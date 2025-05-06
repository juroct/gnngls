# GNNGLS: 图神经网络引导的局部搜索

GNNGLS是一个结合图神经网络(GNN)与传统引导式局部搜索(GLS)的组合优化框架，专注于解决旅行商问题(TSP)及其变体带草稿限制的旅行商问题(TSPDL)。

## 项目特点

- 结合深度学习与传统优化算法
- 支持标准TSP和带草稿限制的TSPDL
- 提供多种算法实现：最近邻、插入、局部搜索、引导局部搜索
- 包含完整的GNN模型训练和应用流程
- 可视化工具展示解决方案和优化过程

## 安装指南

### 环境要求
- Python ≥ 3.10
- PyTorch ≥ 2.2
- DGL ≥ 1.2
- NetworkX ≥ 3.2

### 使用Conda（推荐）

```bash
# 创建并激活conda环境
conda env create -f environment.yml
conda activate gnngls

# 安装包
pip install -e .
```

### 使用pip

```bash
# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 快速入门

### 1. 生成数据集

```bash
# 创建数据目录
mkdir -p data/tspdl/train data/tspdl/val data/tspdl/test

# 生成训练集
python scripts/generate_tspdl_instances.py data/tspdl/train --problem_size 20 --n_instances 100 --hardness medium

# 生成验证集
python scripts/generate_tspdl_instances.py data/tspdl/val --problem_size 20 --n_instances 20 --hardness medium

# 生成测试集
python scripts/generate_tspdl_instances.py data/tspdl/test --problem_size 20 --n_instances 20 --hardness medium

# 创建实例列表
cp data/tspdl/train/instances.txt data/tspdl/train/train.txt
cp data/tspdl/val/instances.txt data/tspdl/val/val.txt
cp data/tspdl/test/instances.txt data/tspdl/test/test.txt
```

### 2. 训练GNN模型

```bash
# 创建模型目录
mkdir -p models/tspdl

# 训练GNN模型
python scripts/train_tspdl_gnn.py data/tspdl models/tspdl --embed_dim 128 --n_layers 3 --n_heads 8 --batch_size 32 --n_epochs 100
```

### 3. 评估和应用

```bash
# 创建结果目录
mkdir -p results/tspdl

# 评估算法性能
python scripts/evaluate_tspdl.py data/tspdl/test results/tspdl --model_path models/tspdl/best_model.pt --time_limit 10.0
```

### 4. 运行演示

```bash
# 运行TSPDL演示
python scripts/tspdl_demo.py demo_output --problem_size 20 --hardness medium --time_limit 5.0
```

## 代码示例

### 在Python代码中使用

```python
import torch
from gnngls import TSPDLInstance, TSPDLEdgeModel, nearest_neighbor_tspdl, gnn_guided_local_search_tspdl

# 创建问题实例
instance = TSPDLInstance.generate_random(problem_size=20, hardness='medium')

# 加载训练好的模型
model = TSPDLEdgeModel(
    in_dim=3,
    embed_dim=128,
    out_dim=1,
    n_layers=3,
    n_heads=8
)
model.load_state_dict(torch.load("models/tspdl/best_model.pt"))
model.eval()

# 生成初始解
initial_tour = nearest_neighbor_tspdl(instance)

# 应用GNN引导的局部搜索
import time
t_limit = time.time() + 5.0  # 5秒时间限制
tour, cost, _ = gnn_guided_local_search_tspdl(
    instance, model, initial_tour, t_limit
)

# 查看解的信息
from gnngls import TSPDLSolution
solution = TSPDLSolution(instance, tour)
print(f"成本: {solution.cost:.4f}, 可行: {solution.feasible}")
```

## 项目结构

- `gnngls/`: 主要包
  - `algorithms.py`: TSP算法实现
  - `models.py`: GNN模型架构
  - `operators.py`: 局部搜索操作符
  - `tspdl.py`: TSPDL问题定义
  - `tspdl_algorithms.py`: TSPDL算法
  - `tspdl_models.py`: TSPDL的GNN模型
  - `tspdl_gnn_gls.py`: TSPDL的GNN引导局部搜索
  - `tspdl_visualization.py`: TSPDL可视化工具
- `scripts/`: 脚本文件
  - `generate_tspdl_instances.py`: 生成TSPDL实例
  - `train_tspdl_gnn.py`: 训练GNN模型
  - `evaluate_tspdl.py`: 评估算法性能
  - `tspdl_demo.py`: TSPDL演示
- `data/`: 数据目录
- `models/`: 模型存储目录
- `docs/`: 文档

## 算法列表

1. **传统算法**
   - 最近邻（Nearest Neighbor）
   - 插入法（Insertion）
   - 局部搜索（Local Search）
   - 引导局部搜索（Guided Local Search）

2. **GNN模型**
   - 边属性预测模型（Edge Property Prediction）
   - 节点属性预测模型（Node Property Prediction）

3. **混合算法**
   - GNN引导的局部搜索（GNN-Guided Local Search）
   - 强化学习方法（Reinforcement Learning）

## 外部求解器

项目支持外部求解器如Concorde和LKH。详细安装和使用说明见[docs/external_solvers.md](docs/external_solvers.md)。

## 许可证

本项目使用MIT许可证，详见[LICENSE](LICENSE)文件。