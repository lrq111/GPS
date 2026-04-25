import logging
import random
import re
import numpy as np
import json
import collections
import string


def compute_answer_f1(a_gold, a_pred):
  """Copied from SQuAD 2.0 evaluation script."""
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def get_tokens(s):
  """Copied from SQuAD 2.0 evaluation script."""
  if not s: return []
  return normalize_answer(s).split()


def normalize_answer(s):
  """Copied from SQuAD 2.0 evaluation script."""
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def extract_nodes_and_edges_loose(text):
                                             
    nodes_match = re.search(r"<nodes>\s*(\[\s*{.*?}\s*\])", text, re.DOTALL)
                                             
    edges_match = re.search(r"<edges>\s*(\[\s*{.*?}\s*\])", text, re.DOTALL)

    if not nodes_match:
        # print("Cannot find <nodes> block!!!\n")
        return None, None
        # raise ValueError("Cannot find <nodes> block")
    if not edges_match:
        # print("Cannot find <edges> block!!!\n")
        return None, None
        # raise ValueError("Cannot find <edges> block")

    try:
        nodes = json.loads(nodes_match.group(1))
        edges = json.loads(edges_match.group(1))
    
    except json.JSONDecodeError as e:
        # print("JSON parsing error: " + str(e))
        return None, None
    
    return nodes, edges