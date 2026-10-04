from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]       #.resolve转化为绝对路径并解析掉相对路径符号；.parents[1]获取当前脚本所在目录的上一级目录

def load_data():
    data = load_wine(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)

    conditions = [df['target'] == 0, df['target'] == 1, df['target'] == 2]
    choice = ['class_1', 'class_2', 'class_3']
    df['class'] = np.select(conditions, choice, default='unknown')
    return df

#测试集划分
def split_data(df):
    X = df.drop(columns=['target', 'class'])  # 产生一个副本
    Y = df.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2026, stratify=Y
    )
    # 索引集合交集是否为空（是否重复索引）
    overlap = X_train.index.intersection(X_test.index)
    assert len(overlap) == 0, "训练集和测试集存在重叠索引"  # condition为True则程序继续，否则报错并抛出message
    return X_train, X_test, Y_train, Y_test

#作图进行数据探索
def plot_eda(Y_train, tr_df, output_dir):
    plt.rc('figure', figsize=[15, 10])
    fig, ax = plt.subplots(2, 2)
    sns.histplot(Y_train, ax=ax[0, 0], bins=3)
    ax[0, 0].set_xticks([0, 1, 2])
    ax[0, 0].set_xticklabels(['class_1', 'class_2', 'class_3'])
    ax[0, 0].set_xlabel('class')
    ax[0, 0].set_title('训练集各类别总数')

    s_1 = tr_df.loc[:, ['alcohol', 'proline', 'class']]
    sns.scatterplot(s_1, ax=ax[0, 1], x='alcohol', y='proline', hue='class')
    ax[0, 1].set_title('alcohol和proline在不同类别间的分布')

    s_2 = tr_df.loc[:, 'alcohol']
    sns.histplot(s_2, ax=ax[1, 0], kde=True)
    ax[1, 0].set_title('alcohol直方图与密度图')

    s_3 = tr_df.loc[:, 'proline']
    sns.histplot(s_3, ax=ax[1, 1], kde=True)
    ax[1, 1].set_title('proline直方图与密度图')

    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
    fig.savefig(output_dir / 'wine数据集探索.png')
    plt.close(fig)

def main():
    output_dir = ROOT / 'reports' / 'wine数据集探索'
    output_dir.mkdir(parents=True, exist_ok=True)           # 自动创建目录

    df = load_data()
    print('特征名：\n', df.columns, '\n 数据类型： \n', df.dtypes, '\n 数据：\n', df,
          '\n 缺失值统计: \n', df.isna().sum())

    X_train, X_test, Y_train, Y_test = split_data(df)
    print("训练特征形状：", X_train.shape)
    print("测试特征形状：", X_test.shape)

    tr_df = df.loc[X_train.index].copy()                   # 基于训练集进行探索，防止数据泄漏
    plot_eda(Y_train, tr_df, output_dir)

if __name__ == "__main__":
    main()


'''
从以上图中初步分析得：
训练部分三类分布较为均衡
alcohol的分布中间多两边少；proline则呈现偏态分布；没有明显的极端值
散点图中三个类别均有重合但区分度仍较为明显
'''