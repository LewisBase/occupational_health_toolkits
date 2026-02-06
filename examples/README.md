# OHTK 使用示例

本目录包含 OHTK (Occupational Health Toolkit) 的使用示例。

## 目录结构

```
examples/
├── data/               # 示例数据文件（不纳入版本控制）
│   └── *.csv          # 放置示例数据
├── models/             # 训练的模型文件（不纳入版本控制）
├── results/            # 分析结果输出（不纳入版本控制）
└── nihl_queue_analysis.py   # NIHL 队列数据分析示例
```

## 示例说明

### 1. NIHL 队列数据分析 (`nihl_queue_analysis.py`)

演示如何使用 ohtk 进行职业性噪声聋（NIHL）的队列数据分析。

**功能包括：**
- 数据加载与预处理
- GEE（广义估计方程）模型分析
- LightGBM 预测模型训练
- Cox 比例风险模型（生存分析）
- 噪声暴露阈值分析
- 分位数回归分析

**使用方法：**

```bash
# 准备数据
# 将队列数据文件放入 examples/data/ 目录
# 文件名: lex_aggregated_by_report_median_queue_data.csv

# 运行全部分析
python nihl_queue_analysis.py --input_path ./data --task all

# 仅运行 GEE 分析
python nihl_queue_analysis.py --input_path ./data --task gee

# 仅运行 LightGBM 模型
python nihl_queue_analysis.py --input_path ./data --task lgbm

# 仅运行阈值分析
python nihl_queue_analysis.py --input_path ./data --task threshold
```

**参数说明：**
- `--input_path`: 输入数据路径，默认 `./data`
- `--output_path`: 输出结果路径，默认 `./results`
- `--models_path`: 模型保存路径，默认 `./models`
- `--task`: 分析任务，可选 `all`, `gee`, `lgbm`, `cox`, `threshold`, `quantile`

**数据格式要求：**

队列数据 CSV 文件需包含以下列：
- `worker_id`: 工人唯一标识
- `creation_date`: 检查日期
- `age`: 年龄
- `sex`: 性别 (0=女, 1=男)
- `NIHL346`: 3、4、6 kHz 平均听阈
- `NIHL1234`: 1、2、3、4 kHz 平均听阈
- `LEX_8h_LEX_40h_median`: 噪声暴露水平 (LAeq)

## 依赖安装

示例需要以下可选依赖：

```bash
# 基础依赖（核心功能）
pip install pandas numpy loguru statsmodels scikit-learn

# LightGBM 预测模型
pip install lightgbm

# SHAP 解释
pip install shap

# Cox 生存分析
pip install lifelines

# GAM 非线性分析
pip install pygam
```

## 注意事项

1. `data/`、`models/`、`results/` 目录下的文件不纳入 Git 版本控制
2. 示例数据较大时，请自行准备或联系数据管理员获取
3. 部分分析功能依赖可选包，请根据需要安装
