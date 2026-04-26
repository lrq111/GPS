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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def group_normalize(a, k, alpha=0.5):
                    
    groups = [a[i:i+k] for i in range(0, len(a), k)]
    
                 
    normalized_groups = []
    for group in groups:
        if len(group) == 0:
            continue
                     
        min_val = min(group)
        max_val = max(group)
                              
        if min_val == max_val:
            normalized_group = [1.0 for _ in group]
        else:
            normalized_group = [(1.0 - alpha*(x/max_val)) for x in group]
        normalized_groups.append(normalized_group)
    
              
    result = [item for group in normalized_groups for item in group]
    
    return result


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, config, data: DataProto, return_dict=True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # correct_score_list, length_score_list, ent_score_list, format_score_list, rule_citation_score_list, dag_consistency_score_list = [],[],[],[],[],[]
        correct_score_list, length_score_list, ent_score_list, format_score_list = [],[],[],[]
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            correct_score = result["correct_score"]
            length_score = result["length_score"]
            ent_score = result["ent_score"]
            format_score = result["format_score"]

            correct_score_list.append(correct_score)
            length_score_list.append(length_score)
            ent_score_list.append(ent_score)
            format_score_list.append(format_score)


            # score: float
            # if isinstance(result, dict):
            #     score = result["score"]
            #     # Store the information including original reward
            #     for key, value in result.items():
            #         reward_extra_info[key].append(value)
            # else:
            #     score = result
            # reward = score

            # if self.overlong_buffer_cfg.enable:
            #     overlong_buffer_len = self.overlong_buffer_cfg.len
            #     expected_len = self.max_resp_len - overlong_buffer_len
            #     exceed_len = valid_response_length - expected_len
            #     overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            #     overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            #     reward += overlong_reward
            #     if self.overlong_buffer_cfg.log:
            #         reward_extra_info["overlong_reward"].append(overlong_reward)
            #         reward_extra_info["overlong"].append(overlong_reward < 0)
            # reward_tensor[i, valid_response_length - 1] = reward


            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print("[prompt]", prompt_str)
            #     print("[response]", response_str)
            #     print("[ground_truth]", ground_truth)
            #     if isinstance(result, dict):
            #         for key, value in result.items():
            #             print(f"[{key}]", value)
            #     else:
            #         print("[score]", score)

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[correct_score]", correct_score)
                print("[raw_length_score]", length_score)
                print("[ent_score]", ent_score)
                print("[format_score]", format_score)
                

        length_max = 2000
        filtered_length_score_list = [x for x in length_score_list if x != length_max]
        if len(filtered_length_score_list) > 0:
            true_length_max = max(filtered_length_score_list)
        else:
            true_length_max = length_max

        for l_id in range(len(length_score_list)):
            if length_score_list[l_id] == length_max:
                length_score_list[l_id] = true_length_max
        
        length_score_list = group_normalize(length_score_list, config.actor_rollout_ref.rollout.n)
        print("After Process Length Rewards are:\n")
        print(length_score_list)
        
        reward_extra_info["correct_score"] = correct_score_list
        reward_extra_info["length_score"] = length_score_list
        reward_extra_info["ent_score"] = ent_score_list
        reward_extra_info["format_score"] = format_score_list

        # total_score = [(c*r+c*l*d+f) for c, f, r, l, d in zip(correct_score_list, format_score_list, length_score_list, rule_citation_score_list, dag_consistency_score_list)]
        total_score = [(e+r)*c+f for c, f, r, e in zip(correct_score_list, format_score_list, length_score_list, ent_score_list)]
        
        for i in range(len(data)):
            reward_tensor[i, valid_response_length - 1] = total_score[i]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

        # if return_dict:
        #     return {
        #         "reward_tensor": reward_tensor,
        #         "reward_extra_info": reward_extra_info,
        #     }
        # else:
        #     return reward_tensor