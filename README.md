# LMForge

LMForge is a benchmark-driven, from-scratch lightweight LLM pretraining system.

This initial source baseline is derived from Stanford University's CS336 Spring
2025 Assignment 1 at commit `314d731892705ac8f9606198d73b0d5168e7b846`.
See [NOTICE](./NOTICE) for provenance and [LICENSE](./LICENSE) for license terms.
The package retains its original `cs336_basics` name during this import task;
the LMForge package migration is a separate Phase 1 change.

## Setup

### Environment
Run Python commands in the WSL Ubuntu distribution. The project uses Python
3.12 and manages its environment with `uv`.

```sh
wsl -d Ubuntu
cd /mnt/e/develop/code/LMForge
```

Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
PYTHONUTF8=1 uv run pytest -q
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

