# Model 目录说明

此目录用于存放**训练好的模型权重文件**，不包含 Python 代码。

## 支持的文件格式

- `.pkl` - Pickle 序列化的 sklearn 模型
- `.pth` / `.pt` - PyTorch 模型权重
- `.onnx` - ONNX 格式模型

## 目录结构变更

原 `ohtk/model/` 目录下的代码已迁移至：

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `model/conv_layer/` | `modeling/conv/` | CNN 模型架构 |
| `model/multi_task/` | `modeling/multi_task/` | 多任务学习模型 |
| `model/tab_transfromer/` | `modeling/transformers/` | Transformer 模型 |
| `model/linear_regression/` | `modeling/custom/` | 自定义模型 |
| `model/train_model.py` | `training/train_model.py` | 训练框架 |
| `model/fit_function/` | `algorithms/fitting/` | 拟合函数 |
| `model/data_mining/` | `algorithms/mining/` | 数据挖掘 |

## 导入方式

### 新方式（推荐）

```python
# 模型架构
from ohtk.modeling import CNNModel, MMoELayer, FTTransformer

# 训练工具
from ohtk.training import train_model, StepRunner, EpochRunner

# 算法工具
from ohtk.algorithms import LAeqFunction, Apriori
```

### 旧方式（已弃用，仍可用）

```python
# 会触发 DeprecationWarning
from ohtk.model import CNNModel, train_model, LAeqFunction
```

## 模型文件命名规范

建议使用以下命名格式：

```
{模型类型}_{任务描述}_{版本}.{扩展名}

例如：
- mmoe_nipts_prediction_v1.pth
- regression_model_for_NIPTS_pred_2023.pkl
- ft_transformer_hearing_loss_v2.onnx
```
