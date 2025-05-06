# 使用外部求解器

GNNGLS项目支持集成外部求解器，如Concorde和LKH，以获取TSP问题的最优解或高质量解。这些求解器可以用于：

1. 生成训练数据
2. 评估启发式算法的性能
3. 获取带有固定边的TSP解

## 安装外部求解器

### 安装Concorde

Concorde是一个用于求解TSP的精确求解器，可以找到最优解。安装步骤如下：

#### 方法1：使用pip安装（推荐）

```bash
pip install concorde-tsp
```

注意：这个包可能在某些平台上安装失败，因为它需要编译C代码。

#### 方法2：手动安装

1. 从[Concorde官网](http://www.math.uwaterloo.ca/tsp/concorde.html)下载源代码
2. 按照说明编译安装
3. 确保可执行文件在系统PATH中

### 安装LKH

LKH是一个用于求解TSP的启发式求解器，可以找到接近最优的解。安装步骤如下：

#### 方法1：使用pip安装（推荐）

```bash
pip install python-lkh
```

注意：这个包可能在某些平台上安装失败，因为它需要编译C代码。

#### 方法2：手动安装

1. 从[LKH官网](http://www.akira.ruc.dk/~keld/research/LKH/)下载源代码
2. 按照说明编译安装
3. 确保可执行文件在系统PATH中

### 安装TSPLIB

TSPLIB是一个用于处理TSP问题实例的库，安装步骤如下：

```bash
pip install tsplib95
```

## 使用外部求解器

### 检查求解器是否可用

```python
from gnngls import SOLVERS_AVAILABLE

if SOLVERS_AVAILABLE:
    print("外部求解器可用")
else:
    print("外部求解器不可用")
```

### 使用Concorde求解TSP

```python
from gnngls import solve_tsp_concorde

# 创建一个TSP实例
coordinates = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# 使用Concorde求解
tour, cost = solve_tsp_concorde(coordinates, verbose=True)

print(f"最优路径: {tour}")
print(f"最优成本: {cost}")
```

### 使用LKH求解TSP

```python
from gnngls import solve_tsp_lkh

# 创建一个TSP实例
coordinates = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# 使用LKH求解
tour, cost = solve_tsp_lkh(coordinates, max_trials=1000, verbose=True)

print(f"最优路径: {tour}")
print(f"最优成本: {cost}")
```

### 使用固定边求解TSP

```python
from gnngls import solve_tsp_with_fixed_edges

# 创建一个TSP实例
coordinates = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# 指定必须包含在解中的边
fixed_edges = [(0, 1), (2, 3)]

# 使用Concorde求解带有固定边的TSP
tour, cost = solve_tsp_with_fixed_edges(
    coordinates, fixed_edges, solver='concorde', verbose=True
)

print(f"最优路径: {tour}")
print(f"最优成本: {cost}")
```

### 获取最优解和成本

```python
from gnngls import get_optimal_tour, get_optimal_cost

# 创建一个TSP实例
coordinates = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# 获取最优解
tour, cost = get_optimal_tour(coordinates, solver='concorde')

print(f"最优路径: {tour}")
print(f"最优成本: {cost}")

# 只获取最优成本
cost = get_optimal_cost(coordinates, solver='lkh')

print(f"最优成本: {cost}")
```

## 在没有外部求解器的情况下使用GNNGLS

如果外部求解器不可用，GNNGLS仍然可以使用其内置的启发式算法：

```python
import networkx as nx
from gnngls import nearest_neighbor, insertion, local_search, guided_local_search, tour_cost

# 创建一个TSP实例
G = nx.complete_graph(10)
# 添加节点和边特征...

# 使用最近邻算法
nn_tour = nearest_neighbor(G, 0)
nn_cost = tour_cost(G, nn_tour)

# 使用插入算法
ins_tour = insertion(G, 0)
ins_cost = tour_cost(G, ins_tour)

# 使用局部搜索
edge_weight, _ = nx.attr_matrix(G, 'weight')
ls_tour, ls_cost, _ = local_search(ins_tour, ins_cost, edge_weight)

# 使用引导局部搜索
import time
t_limit = time.time() + 10.0  # 10秒时间限制
gls_tour, gls_cost, _ = guided_local_search(
    G, ins_tour, ins_cost, t_limit, guides=['weight']
)
```

## 故障排除

### 安装问题

如果安装外部求解器时遇到问题，可以尝试以下方法：

1. 确保系统上安装了C/C++编译器
2. 在Linux上，确保安装了必要的开发包（如build-essential）
3. 尝试手动安装求解器，然后将可执行文件路径添加到系统PATH中

### 运行问题

如果在运行外部求解器时遇到问题，可以尝试以下方法：

1. 确保求解器可执行文件在系统PATH中
2. 检查是否有足够的权限运行求解器
3. 对于大型问题，增加时间限制或使用更简单的问题进行测试

## 参考资料

- [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde.html)
- [LKH Heuristic](http://www.akira.ruc.dk/~keld/research/LKH/)
- [TSPLIB](https://github.com/rhgrant10/tsplib95)
- [Python-Concorde](https://github.com/jvkersch/pyconcorde)
- [Python-LKH](https://github.com/jcmvbkbc/python-lkh)