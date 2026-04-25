import os
import json
from hydra import initialize, compose
from omegaconf import OmegaConf
import re
from collections import defaultdict
from verl.utils.prompts import *
from verl.utils.new_utils import *
from verl.utils.evaluate import *
from verl.utils.new_utils import extract_pattern
import argparse
import random


def get_node_by_id(node_id, node_list):
    for node in node_list:
        try:
            if node['node_id'] == node_id:
                return node
        except:
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

    prompt += "\nPlease output only the number of the best-matching reference answer (1 to {}) or None between <index> </index> tag.".format(len(references))
    return prompt

def answer_if_query_possible(context, question):
    prompt = (
        f"You are given a user's original query and a clarification question. Think step by step to determine whether the clarification question can be answered using ONLY the information in the user's query.\n"
        "If it can be answered, provide a short, concise answer.\n"
        "If it cannot be answered based on the user's query, respond with None.\n"
        "Please output the reasoning process between <think> </think> tag and the answer between <answer> </answer> tag.\n"
    )
    prompt += f"User query: {context}\n"
    prompt += f"Clarification question: {question}\n"
    return prompt

def answer_if_possible(context, question, references):
    n = len(references)
    options_str = "\n".join(f"{i}. {ref.strip()}" for i, ref in enumerate(references, start=1))
    return (
        "You are simulating a real user answering a clarification question. "
        "You MUST answer based ONLY on the user's background context below. "
        "Do NOT use outside knowledge. Do NOT make assumptions beyond what the context explicitly states.\n\n"
        f"User background context:\n{context}\n\n"
        f"Clarification question:\n{question}\n\n"
        f"You must choose EXACTLY ONE of the following options by outputting its number (1-{n}). "
        "If the background context does NOT contain enough information to determine the answer, output 0.\n\n"
        f"Options:\n{options_str}\n\n"
        "Respond in this exact format:\n"
        "<think>Quote the exact sentence(s) from the background context that support your choice, "
        "or explain why the context is insufficient.</think>\n"
        "<answer>N</answer>\n\n"
        f"where N is a single integer in [0, {n}]."
    )

class DAGValidationError(ValueError):
    def __init__(self, error_subtype: str, message: str):
        super().__init__(message)
        self.error_subtype = error_subtype

def translate_dag_validation_detail(msg: str) -> str:
    if not msg:
        return ""
    if not any("\u4e00" <= c <= "\u9fff" for c in msg):
        return msg.strip()

    inner = msg
    if "DAG图构建失败:" in inner:
        inner = inner.split("DAG图构建失败:", 1)[-1].strip()
    elif "DAG图构建失败：" in inner:
        inner = inner.split("DAG图构建失败：", 1)[-1].strip()

    import re

    m = re.match(r"节点key不完整[:：]\s*缺少字段\s*(.+)", inner.strip())
    if m:
        return f"Missing required node field: {m.group(1)}"
    if inner.strip() == "节点key不完整" or inner.strip().startswith("节点key不完整") and "缺少字段" not in inner:
        return "Missing required field(s) in a node"

    m = re.match(r"边key不完整[:：]\s*缺少字段\s*(.+)", inner.strip())
    if m:
        return f"Missing required edge field: {m.group(1)}"
    if inner.strip() == "边key不完整" or (inner.strip().startswith("边key不完整") and "缺少字段" not in inner):
        return "Missing required field(s) in an edge"

    m = re.match(r"结论节点 (\d+) 不能有出边", inner.strip())
    if m:
        return f"Conclusion node {m.group(1)} must not have outgoing edges"

    if inner.strip() in ("图包含环，无法进行拓扑排序", "图包含环，无法计算拓扑序"):
        return "The graph contains a cycle; topological ordering is not possible."

    if inner.startswith("检测到重复出边标签"):
        lines = inner.splitlines()
        out_lines = ["Duplicate outgoing edge labels detected."]
        pat = re.compile(r"节点 (\d+) 的出边标签 '([^']*)' 出现 (\d+) 次")
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            m2 = pat.match(line)
            if m2:
                out_lines.append(
                    f"  Node {m2.group(1)}: outgoing label {m2.group(2)!r} appears {m2.group(3)} times."
                )
            else:
                out_lines.append(f"  {line}")
        return "\n".join(out_lines)

    return inner.strip()

def translate_error_to_english(error_info_dict):
    error_type = error_info_dict.get("error_type", "unknown")
    error_message = error_info_dict.get("error_message", "")

    _STRUCTURED_DAG_TYPES = frozenset(
        {
            "invalid_node_schema",
            "invalid_edge_schema",
            "conclusion_has_outgoing_edge",
            "dag_has_cycle",
            "duplicate_edge_labels",
        }
    )

    type_mapping = {
        "json_parse_error": "JSON parsing error",
        "missing_tags": "Missing tags",
        "dag_construction_error": "DAG construction error",
        "invalid_node_schema": "Invalid node schema",
        "invalid_edge_schema": "Invalid edge schema",
        "conclusion_has_outgoing_edge": "Conclusion has outgoing edge(s)",
        "dag_has_cycle": "DAG has a cycle",
        "duplicate_edge_labels": "Duplicate outgoing edge labels",
        "traversal_error": "Graph traversal error",
        "wrong_final_answer": "Wrong final answer",
        "no_traversal_path": "No traversal path found",
        "unknown": "Unknown error",
    }

    error_type_en = type_mapping.get(error_type, error_type)

    if error_type in _STRUCTURED_DAG_TYPES:
        detail = translate_dag_validation_detail(error_message)
        if detail:
            return f"{error_type_en}: {detail}"
        return error_type_en

    if not error_message or not any('\u4e00' <= char <= '\u9fff' for char in error_message):
        if error_message:
            return f"{error_type_en}: {error_message}"
        else:
            return error_type_en

    message_mapping = {
        "Json解析失败": "JSON parsing failed",
        "未找到node或edge标签": "Missing node or edge tags",
        "DAG图构建失败": "DAG graph construction failed",
        "遍历图失败": "Graph traversal failed",
        "找不到遍历路径": "No traversal path found",
        "最终答案不正确": "Final answer is incorrect"
    }

    error_message_en = message_mapping.get(error_message, None)

    if error_message_en is None:
        if "Json解析失败" in error_message or "JSON解析失败" in error_message:
            if ":" in error_message:
                exception_part = error_message.split(":", 1)[-1].strip()
                error_message_en = f"JSON parsing failed: {exception_part}"
            else:
                error_message_en = "JSON parsing failed"
        elif "未找到" in error_message and ("node" in error_message.lower() or "edge" in error_message.lower()):
            error_message_en = "Missing node or edge tags"
        elif "DAG图构建失败" in error_message or "图构建失败" in error_message:
            if ":" in error_message:
                exception_part = error_message.split(":", 1)[-1].strip()
                error_message_en = f"DAG construction failed: {exception_part}"
            else:
                error_message_en = "DAG construction failed"
        elif "遍历图失败" in error_message or "遍历失败" in error_message:
            if ":" in error_message:
                exception_part = error_message.split(":", 1)[-1].strip()
                error_message_en = f"Graph traversal failed: {exception_part}"
            else:
                error_message_en = "Graph traversal failed"
        elif "找不到遍历路径" in error_message or "无遍历路径" in error_message:
            error_message_en = "No traversal path found"
        elif "最终答案不正确" in error_message or "预测答案" in error_message or "答案不正确" in error_message:
            if "预测答案" in error_message and "正确答案" in error_message:
                error_message_en = error_message
            else:
                error_message_en = f"Final answer is incorrect. {error_message}"
        else:
            error_message_en = error_message

    if error_message_en:
        return f"{error_type_en}: {error_message_en}"
    else:
        return error_type_en

def self_reflection_prompt(background, correct_answer, question, document, previous_graph, error_reasons):
    error_message = "\n".join(f"- {r}" for r in error_reasons)
    previous_output = previous_graph
    original_query = question
    original_passage = document

    prompt = f'''You previously attempted to generate a decision graph (DAG) for a question and document, but encountered an error. Before regenerating, you MUST carefully reflect on what went wrong and identify the root causes.

CRITICAL REFLECTION STEP - Before generating a new DAG, you MUST think deeply about the following questions:

1. **Conditional Path Reasoning Errors**:
   - Did you correctly identify all conditional branches in the passage?
   - Are the conditional questions (Condition nodes) logically sound and correctly extracted from the passage?
   - Do the edge labels properly correspond to the answers of the conditional questions?
   - Are all possible paths from conditions to conclusions correctly represented?
   - REFLECT: Was there a misjudgment in understanding which conditions lead to which conclusions?

2. **Graph Structure Extraction Errors**:
   - Are all nodes properly defined with correct node_id, node_type, node_content, and pre_node_id?
   - Do Conclusion nodes have NO outgoing edges? (This is critical - Conclusion nodes must be terminal)
   - Are the pre_node_id lists accurate? Do they only include DIRECT parent nodes (not higher-level ancestors)?
   - Does the graph form a valid DAG (Directed Acyclic Graph) with no cycles?
   - Are all node_id references in edges valid (both "from" and "to" node_ids exist)?
   - REFLECT: Was there an error in how you structured or extracted the graph topology?

3. **Error Analysis**:
   - Look at the specific error message below and trace back what caused it.
   - Identify which specific part of your previous reasoning or structure was flawed.
   - Be explicit about what you will do differently this time.

IMPORTANT: You MUST NOT repeat the same mistakes! Pay extra attention to the aspects that caused the previous failure. Double-check your logic, verify all structural constraints, and ensure complete correctness before outputting.

Specific error information:
{error_message}

Your previous output:
{previous_output}

Now, regenerate the DAG according to the following requirements with EXTREME CAUTION:

1. Based on the passage, decide whether the user question has multiple possible answers that are only applicable when certain user-specific conditions apply.

2. Then, build a graph (DAG) to represent all possible logical branches. The node and edge of the DAG should be json format as follows:

node format:
{{
"node_id": unique integer ID.
"node_type": either "Condition" or "Conclusion", "Conclusion" nodes must be terminal nodes with no outgoing edges.
"node_content": a clarification question if the current node is Condition node; a statement about the final answer if the current node is Conclusion node.
"pre_node_id": list of direct parent node IDs, if a node has multiple parent nodes, the parent nodes should be in OR relationship. Notice: DO NOT include higher-level predecessors here.
}}

edge format:
{{
"from": the starting node_id of the edge, must be a Condition node.
"to": the ending node_id of the edge.
"label": the label of the edge, should be the answer of the starting Condition node's clarification question.
}}

3. Your output must contain only three parts:

<think>
First, in this section, explicitly state what errors you identified in your previous attempt and how you will avoid them. Then, list all possible answers with corresponding logical branches here.
</think>

<nodes> A list of all nodes. Each node must follow json format above. </nodes>

<edges> A list of all edges. Each edge must follow json format above. </edges>

Notice: Do NOT omit the surrounding square brackets [] in either list. Your output must include the above three parts with complete and properly closed tags: <think>...</think>, <nodes>...</nodes> and <edges>...</edges>.

The passage context is:
{original_passage}

The user question is:
{original_query}

The user's background (conditions that affect which answer applies) is:
{background}

The correct answer expected is:
{correct_answer}

Output:
'''
    return prompt

class ConditionalDAG:
    def __init__(self, query, background, nodes, edges, ablation):
        node_key = ['node_id', 'node_content', 'node_type', 'pre_node_id']
        edge_key = ['from', 'to', 'label']

        for node in nodes:
            for node_k in node_key:
                if node_k not in node:
                    raise DAGValidationError("invalid_node_schema", f"节点key不完整: 缺少字段 {node_k}")
        for edge in edges:
            for edge_k in edge_key:
                if edge_k not in edge:
                    raise DAGValidationError("invalid_edge_schema", f"边key不完整: 缺少字段 {edge_k}")

        self.nodes = {n["node_id"]: n for n in nodes}
        self.edges = edges
        self.query = query
        self.background = background
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
                raise DAGValidationError(
                    "conclusion_has_outgoing_edge",
                    f"结论节点 {node['node_id']} 不能有出边",
                )

        self._check_acyclic()
        self._check_duplicate_edges()

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
            raise DAGValidationError("dag_has_cycle", "图包含环，无法进行拓扑排序")

    def _check_duplicate_edges(self):
        edge_map = defaultdict(list)
        for edge in self.edges:
            from_node = edge["from"]
            label = edge["label"]
            edge_map[from_node].append(label)

        duplicate_edges = []
        for node_id, labels in edge_map.items():
            label_counts = defaultdict(int)
            for l in labels:
                label_counts[l] += 1
            for label, count in label_counts.items():
                if count > 1:
                    duplicate_edges.append((node_id, label, count))

        if duplicate_edges:
            err_msg = "检测到重复出边标签:\n"
            for node_id, label, count in duplicate_edges:
                err_msg += f"  节点 {node_id} 的出边标签 '{label}' 出现 {count} 次\n"
            raise DAGValidationError("duplicate_edge_labels", err_msg)

    def _check_edge_predecessor_consistency(self):
        inconsistent_edges = []
        for edge in self.edges:
            from_id = edge["from"]
            to_id = edge["to"]

            if to_id not in self.nodes:
                continue

            pre_list = self.nodes[to_id].get("pre_node_id", [])
            if from_id not in pre_list:
                inconsistent_edges.append((from_id, to_id))

        if inconsistent_edges:
            msg = "检测到边与节点 pre_node_id 不一致：\n"
            for from_id, to_id in inconsistent_edges:
                msg += f"  边 ({from_id} → {to_id}) 未在节点 {to_id} 的 pre_node_id 中声明。\n"
            raise ValueError(msg)

    def get_avg_path_length(self, node_id):
        if node_id not in self.nodes:
            raise ValueError(f"节点 {node_id} 不存在")

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

        if self.ablation == "no":
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

                current_node_question = current_node["node_content"]

                if current_node["node_type"] == "Conclusion":
                    tmp_traversal_path.append(f"Conclusion: {current_node_question}\n\n")
                    traversal_path.append(tmp_traversal_path)

                    return traversal_path, interaction_turns

                current_cond_ans = self._get_condition_query_answer(current_id, self.query)

                if not current_cond_ans:
                    current_cond_ans = self._get_condition_answer(current_id, self.background)
                    interaction_turns += 1

                    if not current_cond_ans:
                        break

                if current_cond_ans in self.adjacency.get(current_id, {}):
                    next_id = self.adjacency[current_id][current_cond_ans]
                else:
                    next_id = self._get_next_node(current_id, current_cond_ans)

                if next_id is None:
                    break

                tmp_traversal_path.append(f"Conditional Judgment Question: {current_node_question}\nAnswer: {current_cond_ans}\n")
                current_id = next_id
                current_node = self.nodes[current_id]

        return traversal_path, interaction_turns

    def find_random_start_node(self):
        start_candidates = []
        for node in self.nodes.values():
            if not node["pre_node_id"]:
                if "conclusion" in node["node_type"].lower():
                    start_candidates.append((node["node_id"], 0))
                else:
                    start_candidates.append((node["node_id"], self.get_avg_path_length(node["node_id"])))
        if not start_candidates:
            raise ValueError("图中没有找到起始节点（前驱为空的Condition节点）")
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
            raise ValueError("图中没有找到起始节点（前驱为空的Condition节点）")
        start_candidates.sort(key=lambda x: x[1], reverse=True)
        return start_candidates

    def _get_next_node(self, current_id, answer):
        if current_id not in self.adjacency:
            return None
        edge_label_map = list()
        for k in self.adjacency[current_id]:
            edge_label_map.append(k)

        query = self.nodes[current_id]["node_content"]
        match_index_content = get_model_response(match_answer_prompt(query, answer, edge_label_map))
        match_index = extract_pattern(match_index_content, "index")
        if not match_index:
            match_index = match_index_content

        if "none" in match_index.lower():
            return None
        else:
            try:
                return self.adjacency[current_id][edge_label_map[int(match_index)-1]]
            except:
                return None

    def _get_condition_query_answer(self, current_id, query):
        possible_ans_content = get_model_response(answer_if_query_possible(query, self.nodes[current_id]['node_content']))
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
        edge_labels = list(self.adjacency[current_id].keys())
        if not edge_labels:
            return None

        prompt = answer_if_possible(query, self.nodes[current_id]["node_content"], edge_labels)
        for _ in range(3):
            resp = get_model_response(prompt, temperature=0.0)
            raw = (extract_pattern(resp, "answer") or "").strip()
            m = re.search(r"\b(\d+)\b", raw)
            if not m:
                continue
            idx = int(m.group(1))
            if idx == 0:
                return None
            if 1 <= idx <= len(edge_labels):
                return edge_labels[idx - 1]

        return None

    def _is_conclusion(self, node_or_id):
        nid = node_or_id if isinstance(node_or_id, int) else node_or_id["node_id"]
        return "conclusion" in self.nodes[nid]["node_type"].lower()

    def _root_condition_nodes(self):
        return [nid for nid, n in self.nodes.items()
                if (not n["pre_node_id"]) and (not self._is_conclusion(nid))]

    def _topo_order(self):
        from collections import deque
        indeg = {nid: 0 for nid in self.nodes}
        adj = {nid: [] for nid in self.nodes}
        for e in self.edges:
            adj[e["from"]].append(e["to"])
            indeg[e["to"]] += 1
        q = deque([nid for nid, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(self.nodes):
            raise DAGValidationError("dag_has_cycle", "图包含环，无法计算拓扑序")
        return order

    @staticmethod
    def _shannon_entropy(prob_dict, eps=1e-12):
        import math
        H = 0.0
        for p in prob_dict.values():
            if p > eps:
                H -= p * math.log(p)
        return H

    def _forward_uniform(self, normalize_leaf=True):
        import math
        order = self._topo_order()
        Pn = {nid: 0.0 for nid in self.nodes}

        roots = self._root_condition_nodes()
        if roots:
            mass = 1.0 / len(roots)
            for r in roots:
                Pn[r] = mass
        else:
            pass

        for u in order:
            if self._is_conclusion(u):
                continue
            out_map = self.adjacency.get(u, {})
            if not out_map:
                continue
            m = len(out_map)
            p_each = 1.0 / m
            for _, v in out_map.items():
                Pn[v] += Pn[u] * p_each

        Pleaf = {}
        total_leaf = 0.0
        leaf_nodes = [nid for nid in self.nodes if self._is_conclusion(nid)]

        if roots:
            for nid in leaf_nodes:
                Pleaf[nid] = Pn[nid]
                total_leaf += Pn[nid]
        else:
            concl_roots = [nid for nid, n in self.nodes.items()
                           if self._is_conclusion(nid) and (not n["pre_node_id"])]
            if concl_roots:
                total_leaf = 1.0
                for nid in leaf_nodes:
                    Pleaf[nid] = 1.0 / len(concl_roots) if nid in concl_roots else 0.0
            else:
                for nid in leaf_nodes:
                    Pleaf[nid] = 0.0
                total_leaf = 0.0

        if normalize_leaf and total_leaf > 0.0:
            for k in Pleaf:
                Pleaf[k] = Pleaf[k] / total_leaf

        dead_ends = [nid for nid, n in self.nodes.items()
                     if (not self._is_conclusion(nid)) and (nid in self.adjacency and not self.adjacency[nid] or nid not in self.adjacency)]
        unreachable = [nid for nid in self.nodes if Pn[nid] == 0.0]

        return Pn, Pleaf, total_leaf, dead_ends, unreachable

    def compute_eta_uniform(self):
        import math
        Pn, Pleaf, total_leaf_mass, dead_ends, unreachable = self._forward_uniform(normalize_leaf=True)

        H_graph = 0.0
        for nid, n in self.nodes.items():
            if self._is_conclusion(nid):
                continue
            out_map = self.adjacency.get(nid, {})
            if not out_map:
                continue
            H_graph += Pn[nid] * math.log(len(out_map))

        H_leaf = self._shannon_entropy(Pleaf)

        if H_graph <= 1e-12:
            eta = 1.0 if total_leaf_mass > 0 else 0.0
        else:
            eta = max(0.0, min(1.0, H_leaf / H_graph))

        return {
            "eta": eta,
            "H_leaf": H_leaf,
            "H_graph": H_graph,
            "total_leaf_mass": total_leaf_mass,
            "dead_end_count": len(dead_ends),
            "unreachable_count": len(unreachable),
        }
