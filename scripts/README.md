# Scripts Directory

本目录存放常用运行脚本模板，方便直接执行与快速复现。

## 脚本说明

- `run_met.sh`：使用 `config/met_*.yaml` 运行 MET 任务的所有模型
- `run_mre.sh`：使用 `config/mre_*.yaml` 运行 MRE 任务的所有模型
- `download_data.sh`：创建数据目录结构并提示需要准备的文件

## 使用示例

```bash
bash scripts/run_met.sh
bash scripts/run_mre.sh
bash scripts/download_data.sh
```

## 运行前准备

1. 确保已安装依赖：

```bash
pip install -r requirements.txt
```

2. 准备 `data/` 目录中的数据和 `.pt` 文件。
3. 编辑 `config/*.yaml` 中的模型、数据和输出路径。
