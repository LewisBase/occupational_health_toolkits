# 中国工人噪声暴露数据分析实验

本目录包含完整的NIHL/NIPTS预测实验框架，支持多模型、多目标的横向对比。

## 目录结构

```
chinese_worker_experiment/
├── config.yaml                 # 实验配置文件（推荐）
├── run_experiment_new.py       # 新版实验脚本（支持多模型对比）
├── run_experiment.py           # 旧版实验脚本（保留向后兼容）
├── results/                    # 实验结果输出目录
│   ├── processed_data.csv      # 处理后的数据
│   ├── models/                 # 训练好的模型
│   ├── *_report.md            # 各模型报告
│   └── *_comparison_report.md # 横向对比报告
└── README.md                   # 本文件
```

## 快速开始

### 1. 使用YAML配置（推荐）

**修改配置文件** `config.yaml`:

```yaml
experiment_name: "your_experiment_name"

# 配置数据源
data:
  file_path: "examples/data/All_Chinese_worker_exposure_data_0401.xlsx"
  use_staffinfo: true  # 使用StaffInfo构建数据

# 配置模型
models:
  - name: "LightGBM"
    type: "lightgbm"
    params:
      num_leaves: 31
      learning_rate: 0.05
      
  - name: "TabTransformer"
    type: "tabtransformer"
    params:
      dim: 32
      depth: 6

# 配置目标变量
targets:
  - "NIHL_1234"
  - "NIHL_346"
  - "NIPTS_1234"
  - "NIPTS_346"
```

**运行实验**:

```bash
cd examples/chinese_worker_experiment
python run_experiment_new.py
```

### 2. 使用旧版脚本（简单快速）

```bash
cd examples/chinese_worker_experiment
python run_experiment.py
```

## 主要特性

### 1. 多模型支持
- **LightGBM**: 高效梯度提升模型
- **TabTransformer**: 基于Transformer的表格数据模型

### 2. 多目标预测
- **NIHL_1234**: 语音频率听力损失（1000, 2000, 3000, 4000 Hz）
- **NIHL_346**: 高频听力损失（3000, 4000, 6000 Hz）
- **NIPTS_1234**: 语音频率噪声诱导永久性阈移
- **NIPTS_346**: 高频噪声诱导永久性阈移

### 3. ISO 1999 对比
- 支持与 ISO 1999:2013/2023 预测结果对比
- 自动集成到数据处理流程

### 4. 可配置特征工程
- 自动检测分类列
- 可配置缺失值填充
- 可配置特征标准化

### 5. 扩展的评估指标
- RMSE (均方根误差)
- MAE (平均绝对误差)
- MAPE (平均绝对百分比误差)
- MedAE (中位数绝对误差)
- R² (决定系数)
- Explained Variance (可解释方差)

### 6. 横向对比报告
自动生成包含以下内容的对比报告：
- 所有模型在所有目标上的性能对比
- 最佳模型推荐
- 详细的配置信息

## 配置说明

### 数据配置 (data)
- `file_path`: 数据文件路径
- `use_staffinfo`: 是否使用StaffInfo构建（推荐true，充分发挥ohtk工具包）

### 特征配置 (features)
- `feature_columns`: 特征列列表（null则使用默认）
- `categorical_columns`: 分类列列表（null则自动检测）
- `fill_missing`: 是否填充缺失值
- `scale_features`: 是否标准化特征

### 模型配置 (models)
每个模型包含:
- `name`: 模型名称
- `type`: 模型类型（lightgbm, tabtransformer）
- `params`: 模型超参数

### 训练配置 (training)
- `n_folds`: 交叉验证折数
- `test_size`: 测试集比例
- `random_state`: 随机种子
- `task_type`: 任务类型（regression, classification）

### 评估配置 (evaluation)
- `metrics`: 评估指标列表
- `compare_with_iso`: 是否与ISO 1999对比

### 输出配置 (output)
- `dir`: 输出目录
- `save_models`: 是否保存模型
- `generate_report`: 是否生成报告

## 输出说明

### 1. 处理后的数据
`results/processed_data.csv`: 包含所有特征和目标变量的清洗后数据

### 2. 训练好的模型
`results/models/`: 包含所有训练好的模型文件（.pkl格式）

### 3. 特征重要性
`results/models/*_importance.csv`: LightGBM模型的特征重要性

### 4. 单模型报告
`results/*_report.md`: 每个模型的详细训练报告

### 5. 横向对比报告
`results/*_comparison_report.md`: 所有模型的性能对比报告

## 扩展使用

### 添加新模型
在 `config.yaml` 的 `models` 部分添加:

```yaml
models:
  - name: "YourModel"
    type: "your_model_type"
    params:
      param1: value1
      param2: value2
```

### 添加新目标变量
在 `config.yaml` 的 `targets` 部分添加:

```yaml
targets:
  - "NIHL_1234"
  - "YourNewTarget"
```

### 自定义特征列
在 `config.yaml` 的 `features` 部分指定:

```yaml
features:
  feature_columns:
    - "age"
    - "duration"
    - "LAeq"
    # 添加更多特征...
```

## 故障排除

### 1. 数据文件未找到
确保 `config.yaml` 中的 `data.file_path` 路径正确

### 2. StaffInfo构建失败
可以设置 `data.use_staffinfo: false` 回退到简单模式

### 3. TabTransformer训练失败
检查是否安装了PyTorch：`pip install torch`

### 4. 内存不足
- 减少 `training.n_folds` 值
- 减少目标变量数量
- 减少模型数量

## 相关文档

- [ohtk.experiments API文档](../../ohtk/experiments/README.md)
- [实验框架设计](../../docs/experiments_design.md)
- [StaffInfo使用指南](../../docs/staffinfo_guide.md)
