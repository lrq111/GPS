from model_api import *
import os
import json
from hydra import initialize, compose
from omegaconf import OmegaConf
import re
import random
from collections import defaultdict
from oracle_retriever import *
from prompts import *
from new_utils import *
from src.AG.models.vllm_models.qwen_inference_model import VLLMInferenceModel
from src.AG.models.vllm_models.llama_inference_model import LLaMA_VLLMInferenceModel
import argparse


def extract_json_blocks(text, tag="prefix"):
    text = re.sub(r"</?(nodes|edges)>", f"<{tag}>", text)
    parts = re.split(f"<{tag}>", text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 3:
        return []

    target_parts = parts[1:3]

    results = []
    for i, content in enumerate(target_parts):
        cleaned = content
        try:
            parsed = json.loads(cleaned)
            results.append(parsed)
        except json.JSONDecodeError:
            fixed = cleaned
            if not fixed.startswith("["):
                fixed = "[" + fixed
            if not fixed.endswith("]"):
                fixed = fixed + "]"
            try:
                parsed = json.loads(fixed)
                results.append(parsed)
            except json.JSONDecodeError:
                results.append(None)
    
    return results


def extract_pattern(text, tag_name="question"):
    pattern = fr"<{tag_name}>(.*?)</{tag_name}>"
    answer_match = re.search(pattern, text, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        plural_tag = f"{tag_name}s"
        plural_pattern = fr"<{plural_tag}>(.*?)</{plural_tag}>"
        plural_match = re.search(plural_pattern, text, re.DOTALL)
        
        if plural_match:
            answer_content = plural_match.group(1).strip()
        else:
            answer_content = ""
    return answer_content


def get_node_by_id(node_id, node_list):
    for node in node_list:
        try:
            if node['node_id'] == node_id:
                return node
        except:
            print("node error!")
            print(node)
            return None
    return None


def get_edge_content_by_id(from_node_id, to_node_id, edge_list):
    for edge in edge_list:
        try:
            if edge['from'] == from_node_id and edge['to'] == to_node_id:
                return edge['label']
        except:
            return None
    return None


def find_paths_to_roots(node, node_list, edge_list, path=None, paths=None):
    if path is None:
        path = [(node, None)]
    if paths is None:
        paths = []

    if len(node['pre_node_id']) == 0:
        paths.append(list(reversed(path)))
        return paths

    for predecessor_id in node['pre_node_id']:
        predecessor = get_node_by_id(predecessor_id, node_list)
        if predecessor:
            if predecessor not in [p[0] for p in path]:
                edge_label = get_edge_content_by_id(predecessor['node_id'], node['node_id'], edge_list)
                path.append((predecessor, edge_label))
                find_paths_to_roots(predecessor, node_list, edge_list, path, paths)
                path.pop()
    return paths


def extract_decision_process(node_list, edge_list):
    conclusion_decision = dict()
    for node in node_list:
        if node['node_type'] == 'Conclusion' or node['node_type'] == 'conclusion':
            paths = find_paths_to_roots(node, node_list, edge_list)
            conclusion_decision[node['node_id']] = paths
    return conclusion_decision


def match_answer_prompt(question, candidate, references):
    prompt = (
        f"You are given a question, a candidate answer and {len(references)} reference answers.\n\n"
        "Your task is to determine which reference answer is the most semantically similar to the candidate answer.\n"
        "Please follow these rules:\n"
        "- If the candidate answer is semantically close to one of the reference answers, return the index of that reference answer (from 1 to {}).\n".format(len(references))
    )

    prompt += "Question:\n" + question.strip() + "\n"
    prompt += "Candidate answer:\n" + candidate.strip() + "\n"
    prompt += "Reference answers:\n"
    for i, ref in enumerate(references, start=1):
        prompt += f"{i}. {ref.strip()}\n"

    prompt += "\nYou must output the number of the best-matching reference answer (1 to {}) between <index> </index> tag.".format(len(references))
    return prompt


def answer_if_query_possible(context, question):
    prompt = (
        f"You will be given a context and a question. Determine whether the question can be answered using ONLY the information provided in the context.\n"
        "If it can be answered, provide the answer.\n"
        "If it cannot be answered based on the context, respond with None.\n"
        "Please output the answer or None between <answer> </answer> tag.\n"
    )
    prompt += f"\nContext:\n{context}\n"
    prompt += f"\nQuestion:\n{question}\n"
    return prompt


def answer_if_possible(context, question, references):
    prompt = (
        f"You are given a user's background context, a question about the context, and {len(references)} reference answers.\n\n"
        "Your task is to determine which reference answer is most correct based on the context.\n"
        "Please output the most correct reference answer between <answer> </answer> tag.\n"
    )
    prompt += f"\nContext:\n{context}\n"
    prompt += f"\nQuestion:\n{question}\n"
    prompt += "Reference answers:\n"
    for i, ref in enumerate(references, start=1):
        prompt += f"{i}. {ref.strip()}\n"
    return prompt


def answer_if_possible_with_document(context, question, references, document):
    prompt = (
        f"You are given: (1) the user's background context, (2) a relevant document (e.g. official guidance or law), "
        f"(3) a question about the context, and (4) {len(references)} reference answers.\n\n"
        "Your task is to determine which reference answer is most correct.\n"
        "Use the user context for user-specific facts. Use the document to infer the applicable scope or framework "
        "(e.g. jurisdiction or region): if the document clearly specifies its scope "
        "and the user context does not contradict it, assume the scenario falls within that scope when answering.\n"
        "Only choose an answer that contradicts the document's stated scope when the user context clearly indicates otherwise.\n"
        "Please output the most correct reference answer between <answer> </answer> tag.\n"
    )
    prompt += f"\nUser context:\n{context}\n"
    prompt += f"\nRelevant document:\n{document}\n"
    prompt += f"\nQuestion:\n{question}\n"
    prompt += "Reference answers:\n"
    for i, ref in enumerate(references, start=1):
        prompt += f"{i}. {ref.strip()}\n"
    return prompt


def dag_reflection_prompt(original_query, original_passage, previous_output, error_type, error_message):
    if error_type == "json_parse":
        error_description = (
            "Your previous output could not be parsed as valid JSON inside the <nodes> / <edges> tags."
        )
    elif error_type == "dag_construct":
        error_description = (
            "Your previous DAG violated one or more structural rules."
        )
    else:
        error_description = "There was an error in your previous output."

    prompt = f'''You previously attempted to generate a decision graph (DAG) but your output was rejected. Below is the EXACT structural violation found in your previous output — you MUST fix these specific items.

{error_description}

Specific violation(s) detected:
{error_message}

Your previous output (for reference):
{previous_output}

## Fix instructions

- Focus on the violation above. Edit only the offending edges/nodes unless the overall structure is fundamentally broken.
- Do NOT reintroduce the same mistake. If the violation is "Conclusion node X appears as `from`", then node X must NOT appear as `from` in ANY edge in your new output.
- Keep all unrelated edges and nodes the same whenever possible.

## Hard Rules (violating any of these invalidates your output)

1. A `Conclusion` node is TERMINAL. Its `node_id` MUST NOT appear as the `from` field of ANY edge.
2. Only `Condition` nodes may have outgoing edges.
3. The graph must be acyclic.
4. Every `from` and `to` must reference an existing `node_id`.
5. No self-loops.

## Output schema

node format:
{{
"node_id": unique integer ID,
"node_type": either "Condition" or "Conclusion",
"node_content": a clarification question for Condition; an answer statement for Conclusion,
"pre_node_id": list of direct parent node IDs (OR relationship for multiple parents; do NOT include higher-level predecessors).
}}

edge format:
{{
"from": starting Condition node_id,
"to": ending node_id,
"label": answer to the starting Condition's question.
}}

## Output format

Your output must contain exactly four tagged blocks, in order:

<think>
State which specific edge(s) or node(s) from the previous output violated the rule, and how you will fix them — then list the final branches.
</think>

<nodes> JSON list of nodes. </nodes>

<edges> JSON list of edges. </edges>

<check>
- Conclusion node IDs: [list all Conclusion node_ids here]
- Confirm none of them appear as `from` in edges: YES/NO
- Confirm no cycles: YES/NO
- Confirm every edge `from`/`to` references an existing node_id: YES/NO
</check>

Notice: Do NOT omit the surrounding square brackets [] in either JSON list.

The user question is:
{original_query}

The passage context is:
{original_passage}

Output:
'''
    return prompt


def build_violation_report(error_type, exception, nodes, edges, raw_output):
    if error_type == "json_parse":
        msg = str(exception) if exception else "unknown"
        pos = 0
        m = re.search(r"char (\d+)", msg)
        if m:
            pos = int(m.group(1))
        start = max(0, pos - 80)
        end = min(len(raw_output), pos + 80)
        snippet = raw_output[start:end].replace("\n", "\\n")
        return (
            f"JSON parsing error: {msg}. "
            f"Problem area (~char {pos}): ...{snippet}... "
            f"Check that <nodes> and <edges> each contain a single JSON array surrounded by [ ] and that commas / quotes are balanced."
        )

    err_msg = str(exception)
    lines = [f"DAG construction error: {err_msg}"]

    if nodes and edges:
        conclusion_ids = {
            n["node_id"]
            for n in nodes
            if isinstance(n, dict) and "conclusion" in str(n.get("node_type", "")).lower()
        }
        node_ids = {n["node_id"] for n in nodes if isinstance(n, dict) and "node_id" in n}

        bad_from_conclusion = []
        missing_ref = []
        self_loops = []
        for idx, e in enumerate(edges):
            if not isinstance(e, dict):
                continue
            efrom = e.get("from")
            eto = e.get("to")
            if efrom in conclusion_ids:
                bad_from_conclusion.append((idx, e))
            if efrom == eto:
                self_loops.append((idx, e))
            if efrom not in node_ids or eto not in node_ids:
                missing_ref.append((idx, e))

        if bad_from_conclusion:
            lines.append(
                f"Edges where `from` is a Conclusion node (must be removed or changed):"
            )
            for idx, e in bad_from_conclusion:
                lines.append(f"  - edge[{idx}]: {json.dumps(e, ensure_ascii=False)}")

        if self_loops:
            lines.append("Self-loop edges (must be removed):")
            for idx, e in self_loops:
                lines.append(f"  - edge[{idx}]: {json.dumps(e, ensure_ascii=False)}")

        if missing_ref:
            lines.append("Edges referencing non-existent node_ids:")
            for idx, e in missing_ref:
                lines.append(f"  - edge[{idx}]: {json.dumps(e, ensure_ascii=False)}")

        if "cycle" in err_msg.lower():
            lines.append(
                "Graph contains a cycle. Remove back-edges until every path ends at a Conclusion."
            )

    if len(lines) == 1:
        lines.append("Check that every node has node_id/node_type/node_content and every edge has from/to/label.")

    return "\n".join(lines)


def llm_inference(model_name, cfg, prompt, qa_model=None):

    if 'qwen' in model_name:
        print("load qwen model...")
        qa_responses = qa_model.batch_prompt([prompt], **cfg['qa_model']['run']['completion_config'])[0]

    elif 'llama' in model_name:
        print("load llama model...")
        qa_responses = qa_model.batch_prompt([prompt], **cfg['qa_model']['run']['completion_config'])[0]
    
    elif 'qw72b' in model_name:
        qa_responses = call_qwen_api(prompt)
    return qa_responses


class ConditionalDAG:
    def __init__(self, query, background, model_name, config, qa_model, nodes, edges, ablation="no", document=None):
        node_key = ['node_id', 'node_content', 'node_type', 'pre_node_id']
        edge_key = ['from', 'to', 'label']

        for node in nodes:
            for node_k in node_key:
                if node_k not in node:
                    print("节点key不完整!\n")
                    raise ValueError("Node key incomplete")
        for edge in edges:
            for edge_k in edge_key:
                if edge_k not in edge:
                    print("边key不完整!\n")
                    raise ValueError("Edge key incomplete")

        self.nodes = {n["node_id"]: n for n in nodes}
        self.edges = edges
        self.query = query
        self.background = background
        self.document = document
        self.model_name = model_name
        self.config = config
        self.qa_model = qa_model
        self.ablation = ablation

        self.adjacency = {}
        for edge in edges:
            from_node = edge["from"]
            if from_node not in self.adjacency:
                self.adjacency[from_node] = {}
            self.adjacency[from_node][edge["label"]] = edge["to"]
        
        edge_key = ['']
        for node in nodes:
            if "conclusion" in node["node_type"].lower() and node["node_id"] in self.adjacency:
                print(f"结论节点 {node['node_id']} 不能有出边!\n")
                raise ValueError(f"Conclusion node {node['node_id']} cannot have outgoing edges")

        self._check_acyclic()

    
    def _check_acyclic(self):
        from collections import deque

        adj_list = {nid: [] for nid in self.nodes}
        in_degree = {nid: 0 for nid in self.nodes}
        
        for edge in self.edges:
            src = edge["from"]
            dst = edge["to"]
            adj_list[src].append(dst)
            in_degree[dst] += 1

        queue = deque([nid for nid, degree in in_degree.items() if degree == 0])
        processed_count = 0

        while queue:
            current = queue.popleft()
            processed_count += 1
            
            for neighbor in adj_list.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if processed_count != len(self.nodes):
            print("图包含环，无法进行拓扑排序!\n")
            raise ValueError("Graph contains cycles, cannot perform topological sort")

    def get_avg_path_length(self, node_id):
        if node_id not in self.nodes:
            print(f"节点 {node_id} 不存在\n")
            raise ValueError(f"Node {node_id} does not exist")
        
        current_node = self.nodes[node_id]
        if "conclusion" in current_node["node_type"].lower():
            return 0.0
        
        path_lengths = []
        self._dfs_collect_lengths(node_id, 0, path_lengths)
        
        if not path_lengths:
            return 0.0
        
        return sum(path_lengths) / len(path_lengths)
        
    def _dfs_collect_lengths(self, current_node_id, current_length, path_lengths):
        current_node = self.nodes[current_node_id]
        if "conclusion" in current_node["node_type"].lower():
            path_lengths.append(current_length)
            return
        
        if current_node_id in self.adjacency:
            for label, next_node_id in self.adjacency[current_node_id].items():
                self._dfs_collect_lengths(next_node_id, current_length + 1, path_lengths)

    def start_traversal(self, start_node_id=None):
        interaction_turns = 0
        get_next_node_none_count = 0
        
        if not self.ablation == "no":
            start_id_list = start_node_id or self._find_start_node()
        else:
            start_id_list = self.find_random_start_node()
        
        traversal_path = list()
        N = list()
        for item, item_len in start_id_list:
            N.append(self.nodes[item]['node_id'])

        traversal_node_set = set()

        while len(N) != 0:
            current_id = N.pop()
            current_node = self.nodes[current_id]
            
            if current_id not in traversal_node_set:
                tmp_traversal_path = list()
                traversal_node_set.add(current_id)
            else:
                if len(N) != 0:
                    current_id = N.pop()
                    current_node = self.nodes[current_id]
                else:
                    break
            
            while current_node:
                current_node_id = current_node["node_id"]
                print(f"开始遍历节点{current_node_id}!\n")
                current_node_question = current_node["node_content"]
                print(f"当前节点对应的内容为：{current_node_question}\n")
                
                if current_node["node_type"] == "Conclusion":
                    print("成功！遍历到结论节点\n")
                    traversal_path.append(tmp_traversal_path)
                    return traversal_path, interaction_turns, get_next_node_none_count
                
                current_cond_ans = self._get_condition_query_answer(current_id, self.query)
                print(f"根据query获取当前节点的答案为：{current_cond_ans}\n")
                if not current_cond_ans:
                    current_cond_ans = self._get_condition_answer(current_id, self.background)
                    interaction_turns += 1
                    print(f"根据background获取当前节点的答案为：{current_cond_ans}\n")
                    if not current_cond_ans:
                        print(f"主动提问答案不确定！\n")
                        break
                
                next_id = self._get_next_node(current_id, current_cond_ans)
                if next_id is None:
                    get_next_node_none_count += 1
                    print(f"失败！节点 {current_id} 不存在对应 {current_cond_ans} 的分支!\n")
                    break
                print(f"成功！下一个节点为{next_id}\n")
                tmp_traversal_path.append(f"Conditional Judgment Question: {current_node_question}\nAnswer: {current_cond_ans}\n")
                current_id = next_id
                current_node = self.nodes[current_id]
                print(f"当前的遍历路径为:\n{tmp_traversal_path}\n\n")
        
        return traversal_path, interaction_turns, get_next_node_none_count

    
    def find_random_start_node(self):
        start_candidates = []
        for node in self.nodes.values():
            if not node["pre_node_id"]:
                if "conclusion" in node["node_type"].lower():
                    start_candidates.append((node["node_id"], 0))
                else:
                    start_candidates.append((node["node_id"], self.get_avg_path_length(node["node_id"])))
        if not start_candidates:
            print("图中没有找到起始节点（前驱为空的Condition节点）\n")
            raise ValueError("No start node found in graph (Condition node with no predecessors)")
        random.shuffle(start_candidates)
        return start_candidates
    

    def _find_start_node(self):
        start_candidates = []
        for node in self.nodes.values():
            if not node["pre_node_id"]:
                if "conclusion" in node["node_type"].lower():
                    start_candidates.append((node["node_id"], 0))
                else:
                    if self._get_condition_query_answer(node["node_id"], self.query):
                        start_candidates.append((node["node_id"], 1))
                        continue
                    start_candidates.append((node["node_id"], self.get_avg_path_length(node["node_id"])))
        
        if not start_candidates:
            print("图中没有找到起始节点（前驱为空的Condition节点）")
            raise ValueError("No start node found in graph (Condition node with no predecessors)")
        start_candidates.sort(key=lambda x: x[1], reverse=True)
        return start_candidates
    
    def _get_next_node(self, current_id, answer):
        if current_id not in self.adjacency:
            return None
        edge_label_map = list()
        for k in self.adjacency[current_id]:
            edge_label_map.append(k)
        
        query = self.nodes[current_id]["node_content"]
        print(match_answer_prompt(query, answer, edge_label_map))
        match_index_content = call_qwen_api(match_answer_prompt(query, answer, edge_label_map))
        print(f"get next node response: {match_index_content}\n")
        match_index = extract_pattern(match_index_content, "index")
        if not match_index:
            match_index = match_index_content
        
        if "none" in match_index.lower():
            return None
        else:
            try:
                return self.adjacency[current_id][edge_label_map[int(match_index)-1]]
            except:
                print(f"Wrong Match Index: {match_index}\n")
                return None

    def _get_condition_query_answer(self, current_id, query):
        possible_ans_content = call_qwen_api(answer_if_query_possible(query, self.nodes[current_id]['node_content']))
        print(answer_if_query_possible(query, self.nodes[current_id]['node_content']))
        print(f"\nget possible answer based on user query response: {possible_ans_content}\n")
        possible_ans = extract_pattern(possible_ans_content, "answer")
        if not possible_ans:
            possible_ans = possible_ans_content
        
        if "none" in possible_ans.lower():
            return None
        else:
            return possible_ans
    
    def _get_condition_answer(self, current_id, query):
        if current_id not in self.adjacency:
            return None
        edge_label_map = list()
        for k in self.adjacency[current_id]:
            edge_label_map.append(k)

        if self.document:
            prompt = answer_if_possible_with_document(
                query, self.nodes[current_id]['node_content'], edge_label_map, self.document
            )
            print("get possible answer prompt (with document):\n")
            print(prompt.replace(self.document, f"<document omitted: {len(self.document)} chars>"))
        else:
            prompt = answer_if_possible(query, self.nodes[current_id]['node_content'], edge_label_map)
            print("get possible answer prompt:\n")
            print(prompt)
        possible_ans_content = call_qwen_api(prompt)
        print(f"\nget possible answer based on user background response: {possible_ans_content}\n")
        possible_ans = extract_pattern(possible_ans_content, "answer")
        if not possible_ans:
            possible_ans = possible_ans_content

        return possible_ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--model_name", type=str, required=True, help="model_name")
    parser.add_argument("--config", type=str, required=True, help="config")
    parser.add_argument("--dataset", type=str, required=True, help="dataset")
    parser.add_argument("--llm_eval", type=str, default="qw72b", help="llm judger")
    parser.add_argument("--ablation", type=str, default="no", help="ablation")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature")
    parser.add_argument("--data_path", type=str, default="", help="override evaluation data json path")
    parser.add_argument("--documents_path", type=str, default="", help="CondQA documents json path")
    args = parser.parse_args()
    
    T_MAX = 10
    right_num = 0
    success_turns_total = 0
    
    cfg = None
    qa_model = None

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    if "qwen" in args.model_name:
        initialize(config_path="coscript/conf")
        cfg = compose(config_name=args.config)
        cfg = dict(cfg)
        cfg['qa_model']['run']['completion_config']['temperature'] = args.temp
        print(cfg['qa_model']['run']['completion_config'])
        qa_model = VLLMInferenceModel(**cfg['qa_model']['model_config'])
    elif "llama" in args.model_name:
        initialize(config_path="coscript/conf")
        cfg = compose(config_name=args.config)
        cfg = dict(cfg)
        cfg['qa_model']['run']['completion_config']['temperature'] = args.temp
        print(cfg['qa_model']['run']['completion_config'])
        qa_model = LLaMA_VLLMInferenceModel(**cfg['qa_model']['model_config'])

    with open(args.documents_path) as f:
        docs = json.load(f)
    url2doc = {d["url"]: d for _, d in enumerate(docs)}
    
    if args.data_path:
        with open(args.data_path, "r") as f1:
            test_data = json.load(f1)
    elif "dag" in args.dataset:
        data_path = ""
        with open(data_path, "r") as f1:
            test_data = json.load(f1)
    elif "condqa" in args.dataset:
        data_path = ""
        with open(data_path, "r") as f1:
            test_data = json.load(f1)
    elif "sharcqa" in args.dataset:
        data_path = ""
        with open(data_path, "r") as f1:
            test_data = json.load(f1)

    for i,x in enumerate(test_data):
        if "dag" in args.dataset:
            summarized_doc = x['document']
            question = x['question']
            scenario = x['user_info']
            answer = x['answer']
            label = 1
        elif "condqa" in args.dataset:
            summarized_doc = x['document']
            question = x['question']
            scenario = x['user_info']
            answer = x['answer']
            label = x['is_conds']
        elif "sharcqa" in args.dataset:
            summarized_doc = x["document"]
            question = x['question']
            scenario = x['scenario']
            answer = x['answer']
            label = x['is_conds']
        print("\n=======================================================================================\n")
        print(f"=== SAMPLE {i} | doc_chars={len(summarized_doc)} | label={label} ===\n")
        print(f"The document content is:\n{summarized_doc}\n\n")
        print(f"The question is: {question}\n")
        print(f"The user background information is: {scenario}\n")
        
        max_retry = 3
        retry_count = 0
        dag_reasoning_result = None
        node_list = None
        edge_list = None
        dag = None
        error_type = None
        last_error = None
        last_error_exc = None
        node_content = None
        edge_content = None
        
        while retry_count <= max_retry:
            if retry_count == 0:
                dag_reasoning = dag_reasoning_prompt.format(query=question, passage=summarized_doc)
                dag_reasoning_result = llm_inference(args.model_name, cfg, dag_reasoning, qa_model)
                print(f"{i}th test data for dag reasoning result (attempt {retry_count + 1}):\n{dag_reasoning_result}\n")
            else:
                print(f"\n开始第 {retry_count + 1} 次反思重试...\n")
                error_msg = build_violation_report(
                    error_type=error_type,
                    exception=last_error_exc,
                    nodes=node_list,
                    edges=edge_list,
                    raw_output=dag_reasoning_result or "",
                )
                print(f"违规报告:\n{error_msg}\n")

                reflection_prompt = dag_reflection_prompt(
                    original_query=question,
                    original_passage=summarized_doc,
                    previous_output=dag_reasoning_result,
                    error_type=error_type,
                    error_message=error_msg
                )
                dag_reasoning_result = llm_inference(args.model_name, cfg, reflection_prompt, qa_model)
                print(f"{i}th test data for dag reasoning result (reflection attempt {retry_count + 1}):\n{dag_reasoning_result}\n")
            
            node_content = extract_pattern(dag_reasoning_result, "node")
            edge_content = extract_pattern(dag_reasoning_result, "edge")
            
            json_parse_success = False
            try:
                node_list = json.loads(node_content)
                edge_list = json.loads(edge_content)
                json_parse_success = True
                print("JSON解析成功！\n")
            except json.JSONDecodeError as e:
                json_block = extract_json_blocks(dag_reasoning_result)
                
                if len(json_block) == 2 and json_block[0] and json_block[1]:
                    node_list = json_block[0]
                    edge_list = json_block[1]
                    json_parse_success = True
                    print("成功通过extract_json_blocks拯救！\n")
                else:
                    error_type = "json_parse"
                    last_error = f"JSON parsing error: {str(e)}"
                    last_error_exc = e
                    print(f"JSON解析失败: {last_error}\n")
                    if retry_count < max_retry:
                        retry_count += 1
                        continue
                    else:
                        print(f"{i}th data wrong node or edge content! Not json format! (已重试{max_retry}次)\n")
                        break
            
            if json_parse_success:
                try:
                    dag = ConditionalDAG(question, scenario, args.model_name, cfg, qa_model, node_list, edge_list, ablation=args.ablation, document=summarized_doc)
                    print("DAG构建成功！\n")
                    if retry_count > 0:
                        print(f"通过反思重试成功修复！(重试次数: {retry_count})\n")
                    break
                except Exception as e:
                    error_type = "dag_construct"
                    last_error = e
                    last_error_exc = e
                    print(f"DAG构建失败: {str(e)}\n")
                    if retry_count < max_retry:
                        retry_count += 1
                        continue
                    else:
                        print(f"{i}th test data wrong dag construction! (已重试{max_retry}次)\n")
                        break
        
        if dag is None:
            continue

        try:
            traversal, interaction_turns, get_next_node_none_count_sample = dag.start_traversal()
            print(f"{i}th data final traversal path:\n{traversal}\n")
        except Exception as e:
            print(f"{i}th data traversal error!\n")
            continue
        
        if interaction_turns:
            pred_label = 1
        else:
            pred_label = 0

        if pred_label == 1 and label == 1:
            tp += 1
        elif pred_label == 1 and label == 0:
            fp += 1
        elif pred_label == 0 and label == 1:
            fn += 1
        else:
            tn += 1
        
        traversal_content = ""
        
        if traversal:
            for idx, traversal_path in enumerate(traversal):
                for traversal_cond_ans in traversal_path:
                    traversal_content += traversal_cond_ans
            deap = dag_enhanced_answer_prompt.format(question=question, document=summarized_doc, reasoning=traversal_content)
            print(f"\nFinal Prompt:\n{deap.replace(summarized_doc, f'<document omitted: {len(summarized_doc)} chars>')}\n")
            final_answer_content = llm_inference(args.model_name, cfg, deap, qa_model)
            print(f"Final answer content: {final_answer_content}\n")
            final_answer = extract_pattern(final_answer_content, "answer")
            print(f"Prediction: {final_answer}\nGround-Truth: {answer}\n")

            eval_res = llm_inference(args.llm_eval, cfg, verify_prompt.format(question=question, answer_a=answer, answer_b=final_answer), qa_model)
            eval_reasoning = extract_pattern(eval_res, "reasoning")
            eval_final_res = extract_pattern(eval_res, "conclusion")
            print("The evaluation reasoning process is:\n")
            print(eval_reasoning)
            print("The evaluation result is:\n")
            print(eval_final_res)

            if "yes" in eval_final_res.lower():
                print(f"{i}th data correct!\n")
                right_num += 1
                success_turns_total += min(interaction_turns, T_MAX)
            else:
                print(f"{i}th data wrong!\n")
        
        else:
            print(f"{i}th data wrong!\n")
        
        print(f"interaction_turns: {interaction_turns}\n")

    p_success = right_num / len(test_data)
    p_failed = 1.0 - p_success
    mct_success = success_turns_total / right_num if right_num > 0 else 0.0
    success_conditioned_efficiency = p_success * mct_success + p_failed * T_MAX

    print("mean right score (overall acc on final answers):", p_success)
    print(f"wct: {success_conditioned_efficiency:.6f}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_label  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("label metrics (pred_label vs is_conds):")
    print(f"  precision: {precision:.6f}")
    print(f"  recall:    {recall:.6f}")
    print(f"  f1:        {f1_label:.6f}")
