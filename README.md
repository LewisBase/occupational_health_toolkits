# 职业健康大数据智能决策系统 (OHTK)

Occupational Health Toolkits - 面向职业健康领域的科研分析平台

## 简介

OHTK 是一个专为职业健康领域设计的综合性数据分析工具包，采用模块化架构，支持从数据采集、检测分析到诊断决策的全流程处理。系统以 **StaffInfo** 为核心数据模型，整合员工基础信息、健康检查记录、职业危害暴露等多维度数据，为职业病研究提供统一的数据处理和分析框架。

### 核心能力

1. **特定职业病深度研究**
   - 噪声性耳聋（NIHL）的检测、诊断与预测
   - 基于 ISO 1999:2013/2023 标准的 NIPTS 预测
   - 支持 PTA/ABR 听力检测结果的结构化处理

2. **队列数据分析**
   - 纵向队列数据的时间序列特征提取
   - GEE、Cox 比例风险、LightGBM 等多种统计建模方法
   - 阈值分析、分位数回归等高级分析功能

3. **区域流行病学分析**
   - 工厂分布与区域噪声水平分析
   - 人群暴露风险评估与趋势预警
   - 地区职业病爆发趋势研究

4. **模型驱动决策**
   - 多任务学习模型（ESMM/MMoE）联合预测
   - 深度学习与传统机器学习方法融合
   - 可解释性分析（SHAP）支持

### 技术特点

- **面向对象设计**：基于 Pydantic v2 的数据模型，提供类型安全和自动验证
- **工厂模式**：NIPTS 预测器和 NIHL 计算器采用工厂模式，支持灵活扩展
- **统一数据流**：StaffInfo 作为数据入口，支持从原始数据到分析 DataFrame 的完整转换
- **兼容性设计**：支持 ChinaCDC 等多种数据格式的自动映射

## 目录结构

```
OCCUPATIONAL_HEALTH_TOOLKITS/
├── ohtk/
│   ├── constants/                    # 常量定义
│   │   ├── auditory_constants.py     # 听力相关常量
│   │   ├── global_constants.py       # 全局常量
│   │   └── industry_constants.py     # 行业常量
│   │
│   ├── staff_info/                   # 员工信息 [核心模块]
│   │   ├── staff_info.py             # StaffInfo 主类
│   │   ├── staff_basic_info.py       # 基础信息
│   │   ├── staff_health_info.py      # 健康检查信息
│   │   ├── staff_occhaz_info.py      # 职业危害暴露信息
│   │   └── auditory_health_info.py   # 听力健康信息
│   │
│   ├── detection_info/               # 检测信息
│   │   ├── base_result.py
│   │   ├── curve_result.py
│   │   ├── point_result.py
│   │   └── auditory_detection/       # 听力检测
│   │       ├── ABR_result.py
│   │       └── PTA_result.py
│   │
│   ├── diagnose_info/                # 诊断信息
│   │   ├── auditory_diagnose.py      # 听力诊断
│   │   ├── nipts_predictor/          # NIPTS 预测器模块
│   │   │   ├── base.py               # 预测器基类
│   │   │   ├── factory.py            # 预测器工厂
│   │   │   ├── iso_predictors.py     # ISO 标准预测器
│   │   │   ├── ml_predictors.py      # 机器学习预测器
│   │   │   └── dl_predictors.py      # 深度学习预测器
│   │   └── nihl_predictor/           # NIHL 计算器模块
│   │       ├── base.py               # 计算器基类
│   │       ├── factory.py            # 计算器工厂
│   │       └── calculators.py        # 计算器实现
│   │
│   ├── hazard_info/                  # 危害因素信息
│   │   ├── base_hazard.py
│   │   └── noise_hazard.py
│   │
│   ├── modeling/                     # 模型架构
│   │   ├── statistical/              # 统计模型
│   │   │   ├── gee_model.py          # GEE 广义估计方程
│   │   │   └── cox_model.py          # Cox 比例风险模型
│   │   ├── boosting/                 # 梯度提升模型
│   │   │   └── lgbm_nihl.py          # LightGBM NIHL 预测
│   │   ├── multi_task/               # 多任务学习模型
│   │   │   ├── mmoe.py
│   │   │   ├── esmm.py
│   │   │   └── cnn_mmoe.py
│   │   └── transformers/             # Transformer 模型
│   │       ├── ft_transformer.py
│   │       └── tab_transformer_pytorch.py
│   │
│   ├── algorithms/                   # 算法工具
│   │   ├── analysis/                 # 分析工具
│   │   │   ├── threshold_finder.py   # 阈值分析
│   │   │   └── quantile_regression.py # 分位数回归
│   │   ├── fitting/                  # 拟合函数
│   │   │   ├── noise_functions.py    # LAeq 拟合
│   │   │   └── hearing_loss_functions.py
│   │   └── mining/                   # 数据挖掘
│   │       └── association_mining.py
│   │
│   ├── training/                     # 训练框架
│   │   ├── train_model.py            # 通用训练框架
│   │   └── multi_task_trainer.py     # 多任务训练器
│   │
│   └── utils/                        # 工具函数
│       ├── data_helper.py
│       ├── database_helper.py
│       ├── decorators.py
│       ├── plot_helper.py
│       ├── queue_data.py             # 队列数据处理
│       └── pta_correction.py         # PTA 校正与 NIHL 标签转换
│
├── examples/                         # 示例代码
│   ├── nihl_queue_analysis.py        # 队列分析示例
│   └── data/                         # 示例数据
│
├── tests/                            # 单元测试
├── setup.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## 快速开始

### 安装

```bash
pip install -e .
```

### StaffInfo 数据处理工作流

StaffInfo 是 OHTK 的核心数据模型，提供统一的数据加载、处理和转换接口：

```python
from ohtk.staff_info import StaffInfo
import pandas as pd

# 步骤 1: 加载原始数据
df = pd.read_csv("occupational_health_data.csv")

# 步骤 2: 批量加载为 StaffInfo 对象
staff_dict = StaffInfo.load_batch_from_dataframe(df)

# 步骤 3: 计算队列特征（check_order, days_since_first）
StaffInfo.build_queue_features_batch(staff_dict)

# 步骤 4: 转换为分析用 DataFrame
analysis_df = StaffInfo.to_analysis_dataframe_batch(staff_dict)

# 查看帮助信息
StaffInfo.help()
```

### NIPTS 预测

```python
from ohtk.diagnose_info import get_nipts_predictor, NIPTSPredictorFactory

# 使用便捷函数
predictor = get_nipts_predictor("iso1999_2023")
result = predictor.predict(
    LAeq=85,
    age=40,
    sex="M",
    duration=10
)
print(f"NIPTS 预测值: {result.value:.2f} dB")

# 查看可用方法
print(NIPTSPredictorFactory.list_methods())
# ['iso1999_2013', 'iso1999_2023', 'ml_pickle', 'ml_linear', 'dl_torch', 'dl_mmoe']
```

### NIHL 计算与标签转换

```python
from ohtk.diagnose_info import get_nihl_calculator
from ohtk.utils.pta_correction import convert_nihl_to_labels

# 准备听力数据
ear_data = {
    'left_ear_3000': 25.0,
    'left_ear_4000': 30.0,
    'left_ear_6000': 35.0,
    'right_ear_3000': 20.0,
    'right_ear_4000': 25.0,
    'right_ear_6000': 30.0,
}

# NIHL 计算
calculator = get_nihl_calculator("standard")
result = calculator.calculate(ear_data, freq_key="346")
print(f"NIHL (346): {result.value:.2f} dB")

# NIHL 标签转换
# 分类规则: 0-25 正常, 25-40 轻度, 40-55 中度, 55+ 重度
nihl_values = [15, 30, 48, 65]
labels = convert_nihl_to_labels(nihl_values, encoding="categorical")
# 结果: ['正常', '轻度', '中度', '重度']
```

### 队列数据统计分析

```python
from ohtk.modeling.statistical import GEEModel, fit_gee, CoxPHModel
from ohtk.modeling.boosting import LGBMNIHLPredictor

# GEE 分析
gee_model = GEEModel(
    outcome="NIHL346",
    exposure="LAeq",
    covariates=["age", "sex", "duration"]
)
gee_model.fit(analysis_df)
print(gee_model.get_summary())

# LightGBM 预测
lgbm = LGBMNIHLPredictor()
lgbm.train(analysis_df, target="NIHL346")
```

## 模块说明

### 核心模块

| 模块 | 说明 |
|------|------|
| `staff_info` | 员工信息管理，将基础信息、健康记录、职业危害暴露整合至人的维度 |
| `detection_info` | 体检检测结果（PTA、ABR 等）的结构化存储 |
| `diagnose_info` | 基于检测结果的诊断推理，包含 NIPTS/NIHL 预测器 |
| `hazard_info` | 职业危害因素（噪声等）的标准化建模 |

### 分析模块

| 模块 | 说明 |
|------|------|
| `modeling/statistical` | 统计模型（GEE、Cox） |
| `modeling/boosting` | 梯度提升模型（LightGBM） |
| `modeling/multi_task` | 多任务学习（ESMM、MMoE） |
| `algorithms/analysis` | 阈值分析、分位数回归 |

### 工具模块

| 模块 | 说明 |
|------|------|
| `utils/pta_correction` | PTA 年龄校正、NIHL 计算与标签转换 |
| `utils/queue_data` | 队列数据预处理 |

## 运行示例

```bash
# 运行队列分析示例
cd examples
python nihl_queue_analysis.py --task all

# 运行特定分析
python nihl_queue_analysis.py --task gee      # GEE 分析
python nihl_queue_analysis.py --task lgbm     # LightGBM 分析
python nihl_queue_analysis.py --task cox      # Cox 分析
```

## 运行测试

```powershell
# Windows PowerShell
python -m pytest tests/ -v
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.9
- pandas
- numpy
- pydantic >= 2.0
- loguru
- scikit-learn
- statsmodels
- lightgbm
- lifelines

## 许可证

MIT License
