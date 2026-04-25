verify_prompt = """
Given a question, a candidate answer, and a ground truth answer, your task is to determine whether the candidate answer is semantically consistent with the ground truth answer based on the following criteria:

## Semantic Consistency Rules ##
1. If the ground truth answer contains a single definite conclusion, the candidate answer should express the same conclusion.
2. The candidate answer must not introduce any conclusions that contradict the ground truth answer.
3. Ignore non-substantial differences:
- Synonym substitution
- Sentence structure variation

## Output Format ##
Your output should consist of two parts: a reasoning part and a conclusion part. These parts must be enclosed within <reasoning> </reasoning> tags and <conclusion> </conclusion> tags, respectively.
The reasoning part should explain your judgment process. The conclusion part's content is "yes" if two answers are considered semantically consistent, otherwise "no".

The question is: {question}
The ground truth answer is: {answer_a}
The candidate answer is: {answer_b}
"""

dag_reasoning_prompt = '''
Given a user question and a relevant passage that are useful for answering the question, your task is to:

1. Based on the passage, decide whether the user question has multiple possible answers that are only applicable when certain user-specific conditions apply.

2. Then, build a graph (DAG) to represent all possible logical branches.

## Hard Rules (violating any of these invalidates your output)

1. A `Conclusion` node is a TERMINAL node. Its `node_id` MUST NOT appear as the `from` field of ANY edge.
2. Only `Condition` nodes may have outgoing edges. Every edge must start from a `Condition` node.
3. The graph must be acyclic: no node can reach itself by following edges.
4. Every `from` and `to` in an edge MUST reference a `node_id` that exists in your nodes list.
5. Do NOT introduce self-loops (edges where `from == to`).

## Output schema

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

<think> List all possible answers with corresponding logical branches here. </think>

<nodes> A list of all nodes. Each node must follow json format above. </nodes>

<edges> A list of all edges. Each edge must follow json format above. </edges>

Notice: Do NOT omit the surrounding square brackets [] in either list. Your output must include the above three parts with complete and properly closed tags: <think>...</think>, <nodes>...</nodes> and <edges>...</edges>.\n

Now, let's begin:

The user question is:\n
{query}

The passage context is:\n
{passage}

Output:
'''

dag_enhanced_answer_prompt = '''
Given a user question, a relevant document and supplemented user information, please infer and summarize the final answer.

If the known information clearly leads to a single, definite conclusion, output only the conclusion as the final answer without any explanation.

The question is:\n
{question}

The supplemented user information is:\n
{reasoning}

The document is:\n
{document}

Output the final answer between <answer> </answer> tags.
'''
