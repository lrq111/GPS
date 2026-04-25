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
Preprocess the condqa dataset to parquet format
"""

import argparse
import os
import re

import datasets
from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.prompts import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    
    data_source = "condqa"

    train_data_file = "data/condqa/train.json"
    test_data_file = "eval/condqa_dataset/condqa_test_split.json" # change test data here

    train_dataset = load_dataset("json", data_files=train_data_file)["train"]
    print(train_dataset)
    test_dataset = load_dataset("json", data_files=test_data_file)["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            document_raw = example.pop("document")
            answer_raw = example.pop("answer")
            scenario_raw = example.pop("user_info")
            is_conds_raw = example.pop("is_conds")
            
            if not is_conds_raw:
                question_raw = f"{scenario_raw}\nMy question is: {question_raw}"
            
            question = dag_reasoning_prompt.format(query=question_raw, passage=document_raw)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "proactive",
                "reward_model": {"style": "rule", "ground_truth": answer_raw, "scenario": scenario_raw, "is_conds": is_conds_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "scenario": scenario_raw,
                    "is_conds": is_conds_raw,
                    "document": document_raw
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    print(train_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
