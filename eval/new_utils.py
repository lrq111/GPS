from typing import List
import random
import re
import json
import os
import collections
import string


# SPECIAL TOKENS
BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'


def flatten_list(lists):
    return [item for sublist in lists for item in sublist]


# Function to divide prompts into batches of size k
def batch_list(
    lst: List, 
    k: int
    ):
    for i in range(0, len(lst), k):
        yield lst[i:i + k]


# Function to divide list into a list of sublists of size k each
def chunk_list(
    lst: List, 
    k: int
    ) -> List[List]:
    """Splits lst into sublists of size k."""
    return [lst[i:i + k] for i in range(0, len(lst), k)]


# Extract conversation history without instruction tags
def extract_history(
    conversation: str
) -> str:
    conversation = conversation[conversation.find('[/INST]'):].strip()
    conversation = conversation.replace('</s>', '')
    conversation = conversation.replace('[/INST]\n', '\nAI Assistant: ')
    conversation = conversation.replace('[INST]', 'You: ')
    return conversation.strip()


def create_turns(
    conversation: str
    ) -> List[str]:
    delim_1 = EOS_TOKEN + B_INST
    delim_2 = E_INST

    # Escape the delimiters to make them safe for use in a regex pattern
    escaped_delim_1 = re.escape(delim_1)
    escaped_delim_2 = re.escape(delim_2)

    # Create a regex pattern that matches either delimiter
    pattern = f"{escaped_delim_1}|{escaped_delim_2}"
    turns = re.split(pattern, conversation)
    # Note that because of the final E_INST token, there is an additional empty stirng at the end of the list
    # As a result, we return everything before the last 1 elements
    return turns[:-1]


def new_extract_history(
    conversation: str,
    ) -> str:
    conversation = conversation[conversation.find('The initial request is as follows: ') + len('The initial request is as follows: '):]
    conversation = strip_whitespace_around_substring(conversation, B_INST)
    conversation = strip_whitespace_around_substring(conversation, E_INST)
    conversation = strip_whitespace_around_substring(conversation, BOS_TOKEN)
    conversation = strip_whitespace_around_substring(conversation, EOS_TOKEN)
    return conversation


def strip_whitespace_around_substring(s, substring):
    # The pattern looks for the substring followed by any amount of whitespace (\s*)
    # and replaces it with just the substring.
    pattern = r'\s*' + re.escape(substring) + r'\s*'
    return re.sub(pattern, substring, s)

# Shuffle keys and values in a dictionary
def shuffle_dict_values(d):
    random.seed(1)
    
    keys = list(d.keys())
    values = list(d.values())
    rotation = random.randint(1, len(keys) - 1)
    
    # Rotate values by one position to the right
    new_values = values[-rotation:] + values[:-rotation]
    
    # Create a new dictionary by reassigning rotated values to original keys
    new_dict = dict(zip(keys, new_values))
    
    return new_dict


def extract_json_from_string(input_string):
                       
    json_match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
                                     
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            # print(f"Error decoding JSON: {e}")
            return None
    else:
        # print("No JSON found in the input string.")
        return None


def read_json_files_from_folder(folder_path):
    json_data_list = []
    file_info = []
    
                 
    for filename in os.listdir(folder_path):
        parts = filename.split("_")
                 
        if len(parts) != 4 or parts[:2] != ['condqa', 'dag']:
            print("Wrong file part!\n")
            continue
        if filename.endswith('.json'):
            try:
                start_id = int(parts[2])
                end_id = int(parts[3].split(".json")[0])
                # print(f"Start id: {start_id} End id: {end_id}\n")
            except ValueError:
                print(parts)
                print("Wrong start id or end id!\n")
                continue
            file_info.append((start_id, filename))
    
    file_info.sort(key=lambda x: x[0])
    
    for start_id, path in file_info:
        print(path)
        file_path = os.path.join(folder_path, path)
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                json_data_list.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}\n")
    return json_data_list


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


if __name__ == "__main__":
    folder_path = "./deepseek_synthetic_data/condqa"
    result_list = read_json_files_from_folder(folder_path)
    print(f"合并后的列表长度：{len(result_list)}")
    print(result_list[0])