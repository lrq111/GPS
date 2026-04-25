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

import os
import re
from typing import Any, Dict, List, Tuple
import json
from tqdm import tqdm
from verl.utils.dag_reasoning import *
from verl.utils.prompts import *
from verl.utils.evaluate import get_model_response
from verl.utils.new_utils import *


def format_reward(response):
    score = 0.0
    # tags
    # reasoning_res = extract_pattern(response, "think")
    # rules_res = extract_pattern(response, "rules")
    node_res = extract_pattern(response, "node")
    edge_res = extract_pattern(response, "edge")
    if node_res and edge_res:
        score += 0.5
        try:
            node_list = json.loads(node_res)
            edge_list = json.loads(edge_res)
            if node_list and edge_list:
                score += 0.5
        except:
            score += 0.0
    else:
        score += 0.0

    return score


def dag_reward(response, answer, question, scenario, document, is_cond, collect_error_info=False):

    length_max = 2000
    correct_score = 0
    length_score = 0
    error_info = None            

                    
    node_content = extract_pattern(response, "node")
    edge_content = extract_pattern(response, "edge")
    # print(f"The {i}th question is:\n{question}\n\nThe {i}th ground truth answer is:\n{answers[i]}\n\n")
    # print(f"Length of Response:{len(response)}\n\n")
    # print(f"The {i}th Response is:\n{response}\n\n")
    
    if node_content and edge_content:
        try:
            node_list = json.loads(node_content)
            edge_list = json.loads(edge_content)
        except Exception as e:
            error_msg = f"Json解析失败: {str(e)}"
            # print("******************************************************\n")
            # print(f"{error_msg}\nResponse is:{response}")
            # print("******************************************************\n")
            if collect_error_info:
                error_info = {
                    "error_type": "json_parse_error",
                    "error_message": error_msg,
                    "response": response,
                    "node_content": node_content,
                    "edge_content": edge_content
                }
            return correct_score, length_max, 0, error_info
    else:
        error_msg = "未找到node或edge标签"
        if collect_error_info:
            error_info = {
                "error_type": "missing_tags",
                "error_message": error_msg,
                "response": response,
                "node_content": node_content,
                "edge_content": edge_content
            }
        return correct_score, length_max, 0, error_info
        # try:
        #     node_list, edge_list = extract_nodes_and_edges_loose(response)
        # except:
        #     print("******************************************************\n")
                                                                
        #     print("******************************************************\n")
        #     # print(f"The node content is:\n{node_content}\n")
        #     # print(f"The edge content is:\n{edge_content}\n")
        #     return correct_score, length_max, 0
    
    try:
                         
        dag = ConditionalDAG(question, scenario, node_list, edge_list, "no")
    except DAGValidationError as e:
        if collect_error_info:
            error_info = {
                "error_type": e.error_subtype,
                "error_message": str(e),
                "response": response,
                "node_list": node_list,
                "edge_list": edge_list,
            }
        return correct_score, length_max, 0, error_info
    except Exception as e:
        error_msg = f"DAG图构建失败: {str(e)}"
        # print("******************************************************\n")
        # print(f"{error_msg}\nResponse is:\n{response}")
        # print("******************************************************\n")
        if collect_error_info:
            error_info = {
                "error_type": "dag_construction_error",
                "error_message": error_msg,
                "response": response,
                "node_list": node_list,
                "edge_list": edge_list
            }
        return correct_score, length_max, 0, error_info
        
    try:
        traversal, interaction_turns = dag.start_traversal()
    except Exception as e:
        error_msg = f"遍历图失败: {str(e)}"
        # print("******************************************************\n")
        # print(f"{error_msg}\nResponse is:{response}")
        # print("******************************************************\n")
        if collect_error_info:
            error_info = {
                "error_type": "traversal_error",
                "error_message": error_msg,
                "response": response,
                "node_list": node_list,
                "edge_list": edge_list
            }
        return correct_score, length_max, 0, error_info


    try:
        ent = dag.compute_eta_uniform()        
        # print("=============================================================================")
        # print("[Entropy]")
        # print(f"  H_leaf   = {ent['H_leaf']:.6f}")
        # print(f"  H_graph  = {ent['H_graph']:.6f}")
        # print(f"  eta  = {ent['eta']:.6f}")
        # print(f"  total_leaf_mass = {ent['total_leaf_mass']:.6f}")
        # print(f"  dead_end = {ent['dead_end_count']}, unreachable = {ent['unreachable_count']}")
        # print("=============================================================================")
        ent_reward = ent['eta']
    except:
        # print("*******************************************************************************")
                             
        # print("*******************************************************************************")
        ent_reward = 0


                                 
    if traversal:
        # print(f"Format correct Response is:{response}\n")
        traversal_content = ""
        for traversal_idx, traversal_path in enumerate(traversal):
            traversal_content += f"The {traversal_idx}th supplemented user information is:\n"
            for traversal_cond_ans in traversal_path:
                traversal_content += traversal_cond_ans
                                                      
        deap = dag_enhanced_answer_prompt.format(question=question, document=document, reasoning=traversal_content)
        final_answer = extract_pattern(get_model_response(deap), "answer")
        
        # # llm as a judge
        # eval_prompt = llm_eval_prompt.format(question=question, document=document, expected=answers[i], predicted=final_answer)
        # eval_res = get_model_response(eval_prompt)
        # print(f"The {i}th question is:\n{question}\n\nThe {i}th ground truth answer is:\n{answers[i]}\n\n")
        # print(f"The {i}th predicted answer is:\n{final_answer}\n\n")
        # print(f"The {i}th evaluation result is: {eval_res}\n\n")     
        # if "yes" in eval_res.lower():
        #     correct_rewards.append(2.0)
        #     length_rewards.append(float(interaction_turns))
        # else:
        #     correct_rewards.append(score)
        #     length_rewards.append(float(interaction_turns))
        
        eval_prompt = verify_prompt.format(question=question, answer_a=answer, answer_b=final_answer)
        eval_final_res = get_model_response(eval_prompt)
        eval_res = extract_pattern(eval_final_res, "conclusion")
    
        if "yes" in eval_res.lower():
            # print("======================================================\n")
            # print(f"The question is:\n{question}\nThe user scenario is:\n{scenario}\nThe ground truth answer is:\n{answer}\n")
            # print(f"The predicted answer is:\n{final_answer}\n")
            # print(f"The correct Response is:\n{response}\n")
                                                          
            # print("======================================================\n")
            if is_cond == 100: 
                correct_score = 1.0
                length_score = float(interaction_turns)
            elif is_cond == 0 and interaction_turns == 0:
                correct_score = 1.0
                length_score = float(interaction_turns)
            elif is_cond == 1 and interaction_turns > 0:
                correct_score = 1.0
                length_score = float(interaction_turns)
            elif is_cond == 0 and interaction_turns > 0:
                correct_score = 1.0
                length_score = float(interaction_turns)
            elif is_cond == 1 and interaction_turns == 0:
                correct_score = 0.0
                length_score = length_max
            else:
                correct_score = 0.0
                length_score = length_max
            return correct_score, length_score, ent_reward, error_info
        else:
            error_msg = f"最终答案不正确。预测答案: {final_answer}, 正确答案: {answer}"
            # print("******************************************************\n")
            # print(f"The wrong question is:\n{question}\nThe user scenario is:\n{scenario}\nThe ground truth answer is:\n{answer}\n")
            # print(f"The wrong predicted answer is:\n{final_answer}\n")
            # print(f"The wrong Response is:\n{response}\n")
                                                          
            # print(f"The wrong evaluation result is: {eval_res}\n\n")
            # print("******************************************************\n")
            if collect_error_info:
                error_info = {
                    "error_type": "wrong_final_answer",
                    "error_message": error_msg,
                    "response": response,
                    "node_list": node_list,
                    "edge_list": edge_list,
                    "traversal_content": traversal_content,
                    "predicted_answer": final_answer,
                    "correct_answer": answer
                }
            return correct_score, length_max, ent_reward, error_info

                        
    else:
        error_msg = "找不到遍历路径"
        # print("******************************************************\n")
        # print(f"Format correct But No Traversal Response is:{response}\n")
        # print("******************************************************\n")
        if collect_error_info:
            error_info = {
                "error_type": "no_traversal_path",
                "error_message": error_msg,
                "response": response,
                "node_list": node_list,
                "edge_list": edge_list
            }
        return correct_score, length_max, ent_reward, error_info


def compute_score(solution_str, ground_truth, extra_info, collect_error_info=False, **kwargs):
                            
    format_score = format_reward(solution_str)
    question = extra_info["question"]
    scenario = extra_info["scenario"]
    document = extra_info["document"]
    is_conds = extra_info["is_conds"]
    # rule_citation, dag_consistency = rules_reward(solution_str, ground_truth, question, scenario, document, is_conds)
    result = dag_reward(solution_str, ground_truth, question, scenario, document, is_conds, collect_error_info=collect_error_info)
    if len(result) == 4:
        correct_score, length_score, ent_score, error_info = result
    else:
                             
        correct_score, length_score, ent_score = result
        error_info = None
    # return {"correct_score":correct_score, "length_score":length_score, "ent_score":ent_score, "format_score":format_score, "rule_citation_score":rule_citation, "dag_consistency_score":dag_consistency}
    return {"correct_score":correct_score, "length_score":length_score, "ent_score":ent_score, "format_score":format_score, "error_info": error_info}