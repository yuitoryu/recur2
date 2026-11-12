# Recurrent Transformer on LLaMA-Factory

本仓库用 LLaMA-Factory 做因果语言模型的全参数预训练。模型代码支持普通循环、循环输出加权、Full AttnRes 和 Block AttnRes，训练前从 `model.json` 选择变体和结构。

## 安装

在 WSL Ubuntu 26.04 下进入仓库根目录，使用 `uv` 建立独立环境。

```bash
git submodule update --init
bash scripts/setup-wsl.sh
```

子模块固定在 LLaMA-Factory v0.9.5（`7af909522a951e3ad9f022ea6f88b6755257eaa5`）。安装脚本使用 Python 3.11、PyTorch 2.8.0 + CUDA 12.8 和 Transformers 4.57.6，需要支持 CUDA 12.8 的 NVIDIA 驱动。可通过 `RECUR_PYTHON` 指定已有的 Python 3.11 路径，或通过 `UV_BIN` 指定 `uv`。

`requirements-wsl.lock` 记录 Ubuntu 26.04、Python 3.11.18 环境中的实际依赖版本。安装脚本会优先使用这个锁定文件。

仓库位于 Windows 挂载盘上时，如果 `uv` 解包遇到缓存重命名权限错误，可把下载缓存放到 WSL 原生文件系统，虚拟环境仍安装在仓库里。

```bash
UV_CACHE_DIR="$HOME/.cache/uv" UV_LINK_MODE=copy bash scripts/setup-wsl.sh
```

## 初始化和训练

从本仓库根目录下载 FineWeb-Edu 分片、转换数据并初始化模型。可换用自己的 `model.json`。

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
.venv/bin/python -B scripts/download_data.py
.venv/bin/python -m recur_model.convert_data
.venv/bin/python -m recur_model.prepare configs/attnres-50m.json artifacts/attnres-50m --cutoff-len 2048
```

下载脚本默认将最多 20 亿训练 token 和 1000 万验证 token 分别写入 `data/train` 和 `data/valid`。用 `--tokens` 和 `--valid-tokens` 修改下载量，`--valid-tokens 0` 跳过验证集下载。转换脚本默认从 `data/` 读取分片，将结果写入 `artifact/data/fineweb-edu/`。

将以下配置保存为 `train.yaml`，再运行 `.venv/bin/python -m recur_model.train train.yaml`。只转换了训练集时，省略 `eval_dataset` 和 `do_eval`。

```yaml
model_name_or_path: artifacts/attnres-50m
trust_remote_code: true
train_from_scratch: true
stage: pt
finetuning_type: full
do_train: true
do_eval: true
dataset_dir: artifact/data/fineweb-edu
dataset: fineweb_edu_train
eval_dataset: fineweb_edu_valid
cutoff_len: 2048
output_dir: runs/attnres-50m
```

`prepare` 生成包含 `config.json`、模型代码和 GPT-2 tokenizer 的目录。它接受 `--tokenizer` 指定已有的本地 tokenizer。训练入口读取 LLaMA-Factory YAML 和 `key=value` 覆盖项，先校验 `cutoff_len`，再调用 LLaMA-Factory 的训练工作流。已生成的目录也可交给原生 `llamafactory-cli train`。

`recur_pattern` 每项为 `[独立层数, 执行次数]`。同一组的层权重跨轮共享，Full/Block AttnRes 的查询和归一化参数按轮独立。Block AttnRes 的 `block_size` 以 attention/FFN 子层为单位，块边界跨组连续计数。改变结构后，使用新的输出目录生成配置。

示例沿用 50,257 个 GPT-2 token，并复用 EOS 作为 padding。LLaMA-Factory v0.9.5 的 PT collator 会把所有与 pad ID 相同的目标标为忽略，因此此配置的 loss/PPL 不计 EOS。批量生成请使用左 padding 并传入 `attention_mask`。生成会重算完整上下文，使用 `use_cache=False`。

训练保存的 checkpoint 包含用于重建模型的配置和代码。用 `AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)` 直接载入模型。LLaMA-Factory 从 checkpoint 开始一项新训练时，使用 checkpoint 作为 `model_name_or_path` 并设 `train_from_scratch: false`。恢复同一项训练时，保留原 YAML 和训练计划，并用 `resume_from_checkpoint` 指定要恢复的 checkpoint。

## FlashAttention-2

在 WSL 安装与本项目 PyTorch 匹配的官方 wheel，然后在 LLaMA-Factory YAML 中设置 `flash_attn: fa2` 和 `bf16: true`。也可以在 CUDA 上使用 `fp16: true`。

```bash
bash scripts/setup-wsl.sh --fa2
.venv/bin/python -m recur_model.train train.yaml output_dir=runs/attnres-50m-fa2 flash_attn=fa2 bf16=true
```

四种模型的时序自注意力及循环加权的路由层都使用 Transformers 的 FA2 后端。带 padding 时用变长内核保留每条序列的边界，包含全 padding 行的批次回到带掩码的 PyTorch 实现。AttnRes 的跨层聚合沿用原来的计算。

如果模型目录是添加 FA2 前生成的，用原 `model.json` 和当前 `prepare` 生成一个新目录，再用它作为 `model_name_or_path`。继续同一项训练时，可通过 `resume_from_checkpoint` 读取旧 checkpoint 的权重和训练状态。
