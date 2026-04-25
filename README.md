# GPS
The official repository of "GPS: Graph-guided Proactive Information Seeking in Large Language Models"

## Installation

Start from the upstream [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html). For the current sglang/vLLM setup, also install the project requirements:

```bash
pip install -r requirements_sglang.txt
```

The training and evaluation scripts assume CUDA is available. The default Qwen2.5-7B configuration uses 4 GPUs for training and tensor parallel size 2 for vLLM evaluation.

## Data preprocess

The training data is `data/condqa/train.json`. Generate the parquet file with:

```bash
python examples/data_preprocess/condqa.py --local_dir data/condqa
```

The evaluation script uses the JSON files under `eval/condqa_dataset/` by default:

- `dag_test_split.json`
- `condqa_test_split.json`
- `sharc_test_split.json`
- `documents.json`

## Training

The active training entry point is `recipe/dapo/run_dapo_qwen2.5_7b_condqa.sh`.

Before running, edit the script-level `HOME` value near the top of the file or adapt it for your environment. The script derives `RAY_TMPDIR`, `RAY_DATA_HOME`, and the default checkpoint directory from that value.

Run training with explicit data and model paths:

```bash
TRAIN_FILE=/absolute/path/to/train.parquet \
TEST_FILE=/absolute/path/to/test.parquet \
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct \
bash recipe/dapo/run_dapo_qwen2.5_7b_condqa.sh
```

Hydra overrides can be appended after the script command:

```bash
bash recipe/dapo/run_dapo_qwen2.5_7b_condqa.sh \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1
```

## Export Checkpoints

Training writes sharded FSDP checkpoints. Merge an actor checkpoint into a HuggingFace-compatible directory with:

```bash
bash scripts/merge.sh --backend fsdp \
    --local_dir /path/to/ckpts/<project_name>/<exp_name>/global_step_<N>/actor \
    --target_dir /path/to/checkpoints/qwen7b-hf
```

The exported directory is the path used by evaluation and can also be loaded with `transformers` or served by vLLM.

## Evaluation

Point `eval/coscript/conf/qa_model/7b-rl.yaml` to the exported model:

```yaml
model_config:
  model: qwen/Qwen2.5-7B-Instruct
  download_dir: /absolute/path/to/checkpoints/qwen7b-hf
  tensor_parallel_size: 2
```

Then run:

```bash
CUDA_DEVICES=0,1 bash eval/run_dev_dag_qwen_7b.sh
```

Logs are written to `eval/dag_reasoning_reflex_log/`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
li2026gps,
title={{GPS}: Graph-guided Proactive Information Seeking in Large Language Models},
author={Ruiqing Li and Yifeng Xu and Xinke Jiang and Zhibang Yang and Xinyu Ma and Yue Fang and Junfeng Zhao and Yasha Wang and Xu Chu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=xpKe5qMaY4}
}
```

---

## Acknowledgements

This project builds on [verl](https://github.com/volcengine/verl). We thank the verl maintainers and the open-source RL community.
