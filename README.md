# Proto-LoRA

本项目包含用于多模态关系抽取（MRE）和实体对齐/关系抽取任务的 LoRA 微调代码。项目整合了 BLIP2、LLaVA、Qwen2-VL 等模型，并提供训练、验证、测试流程。

> 备注：本仓库已改为配置驱动运行，建议使用 `config/*.yaml` 统一管理路径和超参。

## 目录结构

- `config/`：配置文件目录，支持 `*.yaml` 或 `*.json` 参数化运行
- `data/`：数据准备说明
- `scripts/`：常用运行脚本和下载说明
- `requirements.txt`：Python 依赖列表
- `trainlora_blip2_MET.py`：BLIP2 + MET 训练/验证/测试代码
- `trainlora_llava_MET.py`：LLaVA + MET 训练/验证/测试代码
- `trainlora_Qwen_MET.py`：Qwen2-VL + MET 训练/验证/测试代码
- `trainlora_blip2_MRE.py`：BLIP2 + MRE 训练/验证/测试代码
- `trainlora_llava_MRE.py`：LLaVA + MRE 训练/验证/测试代码
- `trainlora_Qwen2_MRE.py`：Qwen2-VL + MRE 训练/验证/测试代码
- `run_MET.sh`：兼容配置驱动的 MET 运行脚本
- `run_MRE.sh`：兼容配置驱动的 MRE 运行脚本

## 运行环境

建议使用 GPU 环境，并安装如下依赖：

```bash
pip install -r requirements.txt
```

如果需要隔离环境，可以使用 `conda` 或 `venv`：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使用说明

### 1. 修改配置

建议先编辑 `config/*.yaml`，统一管理模型、数据和输出路径，避免直接修改脚本内的路径。

你也可以使用 `scripts/download_data.sh` 创建目录结构并查看数据准备说明。

常用配置项：

- `model_path` / `model_name`
- `prototype_path`
- `train_path`
- `val_path`
- `test_path`
- `image_dir` / `image_base_path`
- `output_dir` / `save_dir`

### 2. 运行单个实验

建议使用配置文件运行：

```bash
bash run_MRE.sh
bash run_MET.sh
```

如果只想执行某一个脚本，可直接运行：

```bash
python trainlora_Qwen2_MRE.py --config config/mre_qwen.yaml
```

或者覆盖配置参数：

```bash
python trainlora_Qwen2_MRE.py \
  --model_name /path/to/Qwen2-VL-7B \
  --prototype_path /path/to/similarity_embedding_Qwen2.pt \
  --train_path /path/to/mnre_llava_train.json \
  --val_path /path/to/mnre_llava_val.json \
  --test_path /path/to/mnre_llava_test.json \
  --image_base_path /path/to/image \
  --save_dir ./outputs/qwen2_mre \
  --epochs 3 \
  --batch_size 1 \
  --lr 2e-5
```

### 3. 输出结果

训练时会保存：

- `best_adapter/`：最佳 LoRA 适配器权重
- `latest_adapter/`：最新一轮 LoRA 适配器权重
- `train_history.json`：训练历史与验证结果
- `test_results.json`：最终测试结果
- `test_results_epoch_X.json`：每轮测试结果
- `run_config.json`：本次运行配置

## 重要说明

- 本项目当前依赖本地数据和模型目录，请在公开仓库中补充数据下载脚本或 Dataset 说明。
- 论文开源仓库应避免硬编码绝对路径，建议统一改成命令行参数或配置文件。
- 如果要公开发布，建议补充 `LICENSE` 文件，并在 README 中说明引用/使用协议。

## 建议补充内容（顶刊/顶会要求）

1. **README 完整性**
   - 任务描述、模型架构、实验设置、数据来源、复现实验步骤
   - 依赖安装、环境配置、GPU/CPU 要求
   - 目录结构说明与关键脚本说明

2. **可复现性**
   - 添加完整的训练/评估复现命令
   - 提供实验配置文件或 `config/*.json` 形式配置
   - 提供数据预处理脚本和数据格式说明
   - 提供 `run_MET.sh` / `run_MRE.sh` 的示例命令

3. **代码质量**
   - 删除本地绝对路径，改成参数或配置
   - 加注释和 docstring，特别是数据加载、指标计算、保存逻辑
   - 把重复逻辑抽成通用模块（例如 `dataset.py`、`utils.py`）

4. **实验结果与可视化**
   - 增加模型性能结果表、对比 Baseline、消融实验
   - 保存 `results/` 示例输出，以及准确率/F1 等指标

5. **附加文件**
   - `LICENSE`（例如 MIT / Apache 2.0）
   - `CONTRIBUTING.md`（可选）
   - `.gitignore`
   - `requirements.txt` / `environment.yml`
   - `CITATION.cff`（可选）

## 版本说明

- 当前仓库页面以 `Proto-LoRA` 为主目录，本次补充了 `README.md` 和 `.gitignore`。
- 你还可以继续补充 `config/`、`scripts/`、`data/` 下载说明，以及更清晰的 `utils/` 模块划分。
