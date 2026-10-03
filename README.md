# ML Foundation

一个用于学习机器学习基础流程的 Python 项目。目前包含 Wine 数据集的读取与训练集划分，以及 Boston Housing 数据的探索性分析。

## 当前内容

- `scr/train.py`：加载 scikit-learn 内置 Wine 数据集，查看特征与数据类型，并按 8:2 划分训练集和测试集。
- `exercises/exercises.py`：读取 Boston Housing CSV，输出字段类型、描述性统计和 `ZN` 列的频数统计。
- `data/Boston Housing Data.csv`：练习使用的数据文件，其中部分字段含缺失值 `NA`。

> 当前项目尚未实现模型训练、模型评估或自动化测试。

## 项目结构

```text
ml-foundation/
├── data/
│   └── Boston Housing Data.csv
├── exercises/
│   └── exercises.py
├── reports/
├── scr/
│   └── train.py
├── environment.yml
└── README.md
```

代码目录名称当前为 `scr`，运行时请勿写成常见的 `src`。

## 环境准备

项目使用 Python 3.13，主要依赖如下：

- pandas
- scikit-learn
- matplotlib
- seaborn

Conda 环境名称为 `MACHINELEARING`（保留项目当前拼写）。如果本机尚未创建该环境，可安装运行现有脚本所需的最小依赖：

```powershell
conda create -n MACHINELEARING -c conda-forge python=3.13 pandas scikit-learn matplotlib seaborn
conda activate MACHINELEARING
```

完整依赖快照见 `environment.yml`。该文件使用 UTF-16 LE 编码，并包含 Windows 平台构建号和本机 `prefix` 路径；部分 Conda 版本无法直接读取，不建议将其作为跨机器安装文件。

## 运行

在仓库根目录执行：

```powershell
conda activate MACHINELEARING
python scr/train.py
python exercises/exercises.py
```

也可以不激活环境，使用以下已验证命令。建议顺序执行，避免 Windows 下并发调用 Conda 时发生临时文件冲突：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
conda run --no-capture-output -n MACHINELEARING python scr/train.py
conda run --no-capture-output -n MACHINELEARING python exercises/exercises.py
```

`PYTHONIOENCODING=utf-8` 和 `--no-capture-output` 用于避免 PowerShell/Conda 捕获中文输出时出现编码错误。

## 数据说明

`exercises/exercises.py` 使用脚本所在位置推导仓库根目录，因此无需从特定工作目录解析 CSV 路径。Pandas 会将 CSV 中的 `NA` 自动识别为缺失值；进行清洗、建模或统计解释时应先检查缺失情况。
