# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.dag_reasoning import self_reflection_prompt, translate_error_to_english
from verl.utils.new_utils import extract_pattern
from verl.workers.reward_manager.dapo import group_normalize
import json

                                                         
LENGTH_SCORE_MAX_PLACEHOLDER = 2000


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    
    def _self_reflection_and_regenerate(self, batch: DataProto, reward_extra_infos_dict: dict):
                                                    
        empty_refl_metrics = {
            "reflection/num_all_wrong_groups": 0,
            "reflection/num_success": 0,
            "reflection/num_attempts_total": 0,
            "reflection/avg_attempts_until_success": 0.0,
            "reflection/success_rate": 0.0,
        }

                    
        if not self.config.get("self_reflection", {}).get("enabled", True):
            return batch, dict(empty_refl_metrics)
        
        n_samples = self.config.actor_rollout_ref.rollout.n
        correct_score_list = reward_extra_infos_dict.get("correct_score", [])
        error_info_list = reward_extra_infos_dict.get("error_info_list", [])
        length_score_list = reward_extra_infos_dict.get("length_score", [])
        ent_score_list = reward_extra_infos_dict.get("ent_score", [])
        format_score_list = reward_extra_infos_dict.get("format_score", [])
                  
        max_reflection_attempts = self.config.get("self_reflection", {}).get("max_attempts", 5)
        
                           
        prompt_uids = batch.non_tensor_batch.get("uid", [])
        prompt_uid2samples = defaultdict(list)
        
        for idx, uid in enumerate(prompt_uids):
            prompt_uid2samples[uid].append(idx)
        
                      
        all_wrong_prompts = []
        for uid, sample_indices in prompt_uid2samples.items():
            if len(sample_indices) != n_samples:
                continue
                                 
            all_wrong = True
            for idx in sample_indices:
                if correct_score_list[idx] > 0:              
                    all_wrong = False
                    break
            if all_wrong:
                all_wrong_prompts.append((uid, sample_indices))
        
        if not all_wrong_prompts:
            return batch, dict(empty_refl_metrics)

        print(f"\n[自我反思] 发现 {len(all_wrong_prompts)} 个全错的prompt组，开始自我反思...\n")

        refl_metrics = dict(empty_refl_metrics)
        refl_metrics["reflection/num_all_wrong_groups"] = len(all_wrong_prompts)
        success_attempt_counts = []                        

                            
        reflection_results = []
        for uid, sample_indices in all_wrong_prompts:
                                                      
            first_idx = sample_indices[0]
            data_item = batch[first_idx]
            
                    
            prompt_str = self.tokenizer.decode(
                data_item.batch["prompts"][:data_item.batch["attention_mask"][:data_item.batch["prompts"].shape[0]].sum()],
                skip_special_tokens=True
            )
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            question = extra_info.get("question", "")
            scenario = extra_info.get("scenario", "")
            document = extra_info.get("document", "")
            
                                                     
            candidates: list[dict] = []
            for idx in sample_indices:
                error_info = error_info_list[idx] if idx < len(error_info_list) else None
                if not error_info:
                    continue
                if "node_list" not in error_info or "edge_list" not in error_info:
                    continue
                try:
                    graph_str = json.dumps(
                        {"nodes": error_info["node_list"], "edges": error_info["edge_list"]},
                        ensure_ascii=False,
                        indent=2,
                    )
                except Exception:
                    graph_str = str({"nodes": error_info.get("node_list"), "edges": error_info.get("edge_list")})
                candidates.append(
                    {
                        "sample_idx": idx,
                        "graph_str": graph_str,
                        "reason_cn": f"{error_info.get('error_type', 'unknown')}: {error_info.get('error_message', '')}",
                        "reason_en": translate_error_to_english(error_info),
                    }
                )

            if not candidates:
                print(f"[自我反思] Prompt {uid} 没有可用的 graph+reason（缺 node_list/edge_list），跳过")
                continue

                                                      
            reflection_success = False
            for cand_i, cand in enumerate(candidates):
                sample_idx = cand["sample_idx"]
                previous_graph_str = cand["graph_str"]
                reason_cn = cand["reason_cn"]
                reason_en = cand["reason_en"]
                print(f"[自我反思] Prompt {uid} 组内样本 {cand_i + 1}/{len(candidates)}（sample_idx={sample_idx}）")
                print(f"[自我反思] 错误信息（中文）: {reason_cn}")

                for attempt in range(max_reflection_attempts):
                    refl_metrics["reflection/num_attempts_total"] += 1
                    print(f"[自我反思] Prompt {uid}, 尝试 {attempt + 1}/{max_reflection_attempts}")

                    reflection_prompt = self_reflection_prompt(
                        background=scenario,
                        correct_answer=ground_truth,
                        question=question,
                        document=document,
                        previous_graph=previous_graph_str,
                        error_reasons=[reason_en] if reason_en else ["All sampled generations failed to get correct scores"],
                    )

                                                               
                    max_prompt_length = self.config.data.get("max_prompt_length", 2048)
                    pad_token_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    )
                    messages = [{"role": "user", "content": reflection_prompt}]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,                                     
                    )
                    encoded = self.tokenizer(
                        text,
                        return_tensors="pt",
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_prompt_length,
                    )
                    input_ids = encoded["input_ids"]  # (1, L)
                    attn_mask = encoded.get("attention_mask", (input_ids != pad_token_id).long())
                    seq_len = input_ids.shape[1]
                    if seq_len < max_prompt_length:
                        pad_len = max_prompt_length - seq_len
                        input_ids = torch.cat(
                            [
                                torch.full((1, pad_len), pad_token_id, dtype=input_ids.dtype),
                                input_ids,
                            ],
                            dim=1,
                        )
                        attn_mask = torch.cat(
                            [
                                torch.zeros(1, pad_len, dtype=attn_mask.dtype),
                                attn_mask,
                            ],
                            dim=1,
                        )
                    else:
                        pad_len = 0

                    position_ids = torch.cat(
                        [
                            torch.zeros(1, pad_len, dtype=torch.long),
                            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
                        ],
                        dim=1,
                    )

                                                                                         
                    rollout_cfg = self.config.actor_rollout_ref.rollout
                    trainer_cfg = getattr(self.config, "trainer", None)
                    tp_size = int(getattr(rollout_cfg, "tensor_model_parallel_size", 1) or 1)
                    dp_size = 1
                    if trainer_cfg is not None:
                        try:
                            n_gpus = int(getattr(trainer_cfg, "n_gpus_per_node", 1) or 1)
                            dp_size = max(1, n_gpus // tp_size)
                        except Exception:
                            dp_size = 1

                    if dp_size > 1:
                        input_ids = input_ids.repeat(dp_size, 1)
                        attn_mask = attn_mask.repeat(dp_size, 1)
                        position_ids = position_ids.repeat(dp_size, 1)

                    refl_batch = DataProto.from_dict(
                        tensors={
                            "input_ids": input_ids,
                            "attention_mask": attn_mask,
                            "position_ids": position_ids,
                        },
                        non_tensors={},
                        meta_info={
                            "eos_token_id": self.tokenizer.eos_token_id,
                            "pad_token_id": pad_token_id,
                            "do_sample": True,
                            "validate": False,
                            "temperature": self.config.get("self_reflection", {}).get("actor_temperature", 0.7),
                            "top_p": getattr(rollout_cfg, "top_p", 1.0),
                            "top_k": getattr(rollout_cfg, "top_k", -1),
                        },
                    )
                    refl_out = self.actor_rollout_wg.generate_sequences(refl_batch)
                    refl_ids = refl_out.batch["responses"][0]
                    reflection_response = self.tokenizer.decode(refl_ids, skip_special_tokens=True)

                    print(f"[自我反思] 反思结果（尝试 {attempt + 1}）:")
                    print("-" * 80)
                    print(reflection_response)
                    print("-" * 80)

                                                                            
                    new_node_content = extract_pattern(reflection_response, "nodes") or extract_pattern(
                        reflection_response, "node"
                    )
                    new_edge_content = extract_pattern(reflection_response, "edges") or extract_pattern(
                        reflection_response, "edge"
                    )

                    if not new_node_content or not new_edge_content:
                        print(f"[自我反思] 尝试 {attempt + 1} 失败：未找到nodes/edges标签")
                        continue

                    try:
                        json.loads(new_node_content)
                        json.loads(new_edge_content)
                    except Exception as e:
                        print(f"[自我反思] 尝试 {attempt + 1} 失败：JSON解析错误 - {e}")
                        continue

                    new_response = reflection_response

                    from verl.utils.reward_score.condqa import compute_score

                    new_result = compute_score(
                        solution_str=new_response,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        collect_error_info=False,
                    )

                    if new_result.get("correct_score", 0) > 0:
                        refl_metrics["reflection/num_success"] += 1
                        success_attempt_counts.append(attempt + 1)
                        print(
                            f"[自我反思] 成功！使用 sample_idx={sample_idx} 的 graph+reason 反思得到正确 response，"
                            f"correct_score={new_result.get('correct_score')}"
                        )
                        reflection_results.append(
                            {
                                "uid": uid,
                                "original_indices": sample_indices,
                                "new_response": new_response,
                                "new_result": new_result,
                                "reflection_attempts": attempt + 1,
                                "first_idx": sample_indices[0],              
                                "source_sample_idx": sample_idx,
                            }
                        )
                        reflection_success = True
                        break
                    else:
                        print(f"[自我反思] 尝试 {attempt + 1} 生成的response仍然不正确")

                if reflection_success:
                    break

            if not reflection_success:
                print(f"[自我反思] Prompt {uid} 逐个反思 {len(candidates)} 个样本后仍未生成正确的response")
        
                                                    
        if reflection_results:
            print(f"\n[自我反思] 成功反思 {len(reflection_results)} 个prompt，更新训练batch\n")
            
                                                                                             
                                                         
            reflection_done = []  # [(first_idx, new_response_len), ...]
            for result in reflection_results:
                first_idx = result["first_idx"]
                new_result = result["new_result"]
                new_response = result["new_response"]
                
                                                                                       
                if first_idx < len(correct_score_list):
                    correct_score_list[first_idx] = new_result["correct_score"]
                    if first_idx < len(length_score_list):
                        length_score_list[first_idx] = new_result["length_score"]
                    if first_idx < len(ent_score_list):
                        ent_score_list[first_idx] = new_result["ent_score"]
                    if first_idx < len(format_score_list):
                        format_score_list[first_idx] = new_result["format_score"]
                
                data_item = batch[first_idx]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                new_response_ids = self.tokenizer.encode(
                    new_response,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                if new_response_ids.dim() > 1:
                    new_response_ids = new_response_ids.squeeze(0)
                if new_response_ids.dim() == 0:
                    new_response_ids = new_response_ids.unsqueeze(0)
                
                max_response_len = batch.batch["responses"].shape[1]
                new_response_len_raw = new_response_ids.shape[0]
                if new_response_len_raw > max_response_len:
                    new_response_ids = new_response_ids[:max_response_len].clone()
                    new_response_len = max_response_len
                else:
                    new_response_len = new_response_len_raw
                if new_response_len > 0:
                    device = batch.batch["responses"].device
                    new_response_ids = new_response_ids.to(device)
                    padding_length = max_response_len - new_response_len
                    padded_response = torch.cat([
                        new_response_ids,
                        torch.zeros(padding_length, dtype=torch.long, device=device)
                    ])
                    new_attention_mask = torch.cat([
                        torch.ones(prompt_length + new_response_len, dtype=torch.long, device=device),
                        torch.zeros(padding_length, dtype=torch.long, device=device)
                    ])
                    if new_attention_mask.shape[0] < batch.batch["attention_mask"].shape[1]:
                        new_attention_mask = torch.nn.functional.pad(
                            new_attention_mask,
                            (0, batch.batch["attention_mask"].shape[1] - new_attention_mask.shape[0]),
                            value=0
                        )
                    batch.batch["responses"][first_idx] = padded_response
                    batch.batch["attention_mask"][first_idx] = new_attention_mask
                    prompt_ids_squeezed = prompt_ids.squeeze(0)
                    if prompt_ids_squeezed.device != device:
                        prompt_ids_squeezed = prompt_ids_squeezed.to(device)
                    new_input_ids = torch.cat([prompt_ids_squeezed, new_response_ids])
                    if new_input_ids.shape[0] <= batch.batch["input_ids"].shape[1]:
                        padding_length_input = batch.batch["input_ids"].shape[1] - new_input_ids.shape[0]
                        padded_input_ids = torch.cat([
                            new_input_ids,
                            torch.zeros(padding_length_input, dtype=torch.long, device=device)
                        ])
                        batch.batch["input_ids"][first_idx] = padded_input_ids
                    reflection_done.append((first_idx, new_response_len))
                else:
                    print(f"[自我反思] 警告：样本 {first_idx} 的反思 response 为空，跳过 token_level_scores 更新")
            
                                                                            
            filtered_length = [x for x in length_score_list if x != LENGTH_SCORE_MAX_PLACEHOLDER]
            true_length_max = max(filtered_length) if filtered_length else LENGTH_SCORE_MAX_PLACEHOLDER
            for l_id in range(len(length_score_list)):
                if length_score_list[l_id] == LENGTH_SCORE_MAX_PLACEHOLDER:
                    length_score_list[l_id] = true_length_max
            length_score_list = group_normalize(length_score_list, n_samples)
            
                                                                
            scores_t = batch.batch["token_level_scores"]
            for first_idx, new_response_len in reflection_done:
                new_total = (
                    (ent_score_list[first_idx] + length_score_list[first_idx])
                    * correct_score_list[first_idx]
                    + format_score_list[first_idx]
                )
                scores_t[first_idx, new_response_len:] = 0
                scores_t[first_idx, new_response_len - 1] = float(new_total)
                print(f"[自我反思] 已更新样本 {first_idx} 的 response、分数与 token_level_scores（已按组归一化 length）")
        
                                                            
        reward_extra_infos_dict["correct_score"] = correct_score_list
        reward_extra_infos_dict["length_score"] = length_score_list
        reward_extra_infos_dict["ent_score"] = ent_score_list
        reward_extra_infos_dict["format_score"] = format_score_list

                
        if success_attempt_counts:
            refl_metrics["reflection/avg_attempts_until_success"] = (
                sum(success_attempt_counts) / len(success_attempt_counts)
            )
        if refl_metrics["reflection/num_all_wrong_groups"] > 0:
            refl_metrics["reflection/success_rate"] = (
                refl_metrics["reflection/num_success"]
                / refl_metrics["reflection/num_all_wrong_groups"]
            )

        return batch, refl_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            # pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_result = self.reward_fn(self.config, new_batch, return_dict=True)
                            reward_baseline_tensor = reward_baseline_result["reward_tensor"].sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            # reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_result = self.reward_fn(self.config, new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            # print(f"Error in reward_fn: {e}")
                                                     
                            try:
                                reward_tensor = self.reward_fn(self.config, new_batch, return_dict=False)
                                reward_extra_infos_dict = {}
                            except:
                                               
                                reward_tensor = self.reward_fn(self.config, new_batch)
                                reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                                                      
                        if self.config.get("self_reflection", {}).get("enabled", True):
                            with marked_timer("self_reflection", timing_raw, "purple"):
                                new_batch, refl_metrics = self._self_reflection_and_regenerate(
                                    new_batch, reward_extra_infos_dict
                                )
                            metrics.update(refl_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            # print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                # print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    # pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
