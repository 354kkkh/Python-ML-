# Wine最终评估：冻结配置

> 冻结日期：2026-10-29。
> 决定依据：本人在本次对话中明确确认“采用逻辑回归的Pipeline”，由助手据现有代码和报告填写本记录。
> 状态：方案已确定，配置在最终测试前冻结。本记录创建时，最终评估尚未执行；本次填写未训练最终模型、未预测或评分最终36条测试数据。

## 1. 选定方案与理由

采用`StandardScaler`＋`LogisticRegression(max_iter=1000)`组成的Pipeline，沿用已完成开发CV时的配置，不新增参数搜索或其他模型。

同一142条开发数据、同一五折划分下，当前开发结果为：

| 方案 | Accuracy均值 | 折间标准差 | Macro-F1均值 | 折间标准差 |
|---|---:|---:|---:|---:|
| Dummy | 0.4012 | 0.0103 | 0.1909 | 0.0035 |
| 标准化逻辑回归 | 0.9719 | 0.0259 | 0.9721 | 0.0261 |
| 深度3树 | 0.9296 | 0.0491 | 0.9312 | 0.0475 |
| 不限深度树 | 0.9153 | 0.0473 | 0.9185 | 0.0466 |

逻辑回归的主指标Macro-F1均值较高，折间波动小于两棵树，因此选用当前配置进行一次最终评估。以上仅为固定五折的开发结果，不能证明稳定泛化优势；折间标准差采用`ddof=0`，不代表置信区间，也不用于预估或承诺最终测试成绩。

依据：[四方案开发CV脚本](../../scr/cv_compare_models.py)、[折指标](../cv_model_comparison/fold_metrics.csv)、[开发CV汇总](../cv_model_comparison/summary.csv)。

## 2. 数据、特征与固定划分

- 来源：`sklearn.datasets.load_wine(as_frame=True)`，共178条样本、13个数值特征、3个类别。
- 目标列：`target`，类别为`[0, 1, 2]`。
- 排除：`target`和用于展示的辅助列`class`均不作为输入；样本原索引只用于追查，不作为特征。
- 沿用[train.py](../../scr/train.py)中`load_data()`与`split_data()`的既有逻辑。
- 第一层划分：`test_size=0.2, random_state=2026, stratify=y`，保留原142条开发数据、36条最终测试数据，不更换随机种子或重新挑选成员。
- 原113／29划分和五折CV均为开发阶段实验；最终模型将使用全部142条开发数据重新拟合，最终36条仅用于评价该冻结方案。
- 特征与标签按原索引对齐，开发与最终测试索引互斥；保留13列原始顺序，不新增特征工程、删列或重排。

固定特征顺序如下：

```text
1.  alcohol
2.  malic_acid
3.  ash
4.  alcalinity_of_ash
5.  magnesium
6.  total_phenols
7.  flavanoids
8.  nonflavanoid_phenols
9.  proanthocyanins
10. color_intensity
11. hue
12. od280/od315_of_diluted_wines
13. proline
```

## 3. Pipeline与超参数

沿用四方案CV中逻辑回归的构造方式和步骤名：

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000)),
])
```

本次只读取新建对象的`get_params()`核对配置，没有调用`fit`或`predict`。在当前scikit-learn 1.9.1环境中，实际参数为：

```json
{
  "scaler": {
    "copy": true,
    "with_mean": true,
    "with_std": true
  },
  "classifier": {
    "C": 1.0,
    "class_weight": null,
    "dual": false,
    "fit_intercept": true,
    "intercept_scaling": 1,
    "l1_ratio": 0.0,
    "max_iter": 1000,
    "n_jobs": null,
    "penalty": "deprecated",
    "random_state": null,
    "solver": "lbfgs",
    "tol": 0.0001,
    "verbose": 0,
    "warm_start": false
  }
}
```

参数快照按当前环境原样记录；运行时沿用上方Pipeline构造代码。其中分类器`random_state=None`是既有默认值，与数据划分的`random_state=2026`分属不同设置，不为追分更改。

冻结的是模型类型、预处理方式及超参数。最终训练时会新建Pipeline，在142条开发数据上拟合一次；Scaler的均值、标准差及逻辑回归权重由这142条学习，不使用某个CV折模型或在178条数据上学习的统计量替代。

## 4. 最终评价规则与产出

- 主指标：Macro-F1，`average='macro', labels=[0,1,2]`。
- 辅助指标：Accuracy。
- 混淆矩阵：`labels=[0,1,2]`，行是真实类别、列是预测类别，元素总和为36。
- 对原36条最终测试使用同一份预测计算上述结果，不为不同指标额外训练模型。
- 不设最低分数门槛，也不要求测试分数超过CV均值。CV和最终测试的数据成员、训练样本数与用途不同，分数差异如实解释。

后续任务5计划生成：

| 文件 | 内容 |
|---|---|
| `reports/final_evaluation/metrics.csv` | 1行，`model、split、n_samples、accuracy、macro_f1`；`split='final_test'`，样本数36 |
| `reports/final_evaluation/test_predictions.csv` | 36行，`sample_index、y_true、y_pred`，保持原测试顺序，原索引唯一 |
| 本周笔记或矩阵文本 | 混淆矩阵、实际结果、局限及实际用时 |

本记录不包含最终测试成绩；上述产出由后续实际运行生成。

## 5. 结果使用约定

1. 正式最终评价只针对本记录冻结的方案，成绩无论高低都保留并如实报告。
2. 查看测试结果后，不据此换模型、改参数、换种子、换测试成员或补测其他候选来挑最高分。
3. 对同一冻结方案的完全相同复现需明确标为复核，不把它变成新一轮模型选择。
4. 若因测试结果继续改动方案，应说明原测试数据已参与开发，后续同一测试集的结果不能再当作独立最终评价。
5. 实现出错时先保存完整报错、已执行步骤及是否已查看测试结果，再核对代码与冻结记录；不因调试随意改变模型配置或倒写本记录。

## 6. 冻结时的环境与代码版本

- Conda环境：`ML`。
- 解释器：`D:\conda_envs\ML\python.exe`。
- Python：3.14.8。
- scikit-learn：1.9.1。
- 本地代码基准提交：`c8a1a389f2aaf84635f902ddc8616af54026ba4a`。
- 冻结记录由本次单独新增，不声称已包含在上述提交中；本次未执行暂存、提交或推送。
- 依赖快照：[environment.yml](../../environment.yml)，为本机Windows快照；全新机器环境重建未验证。
