import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def na_count(series):
    s = series.isnull().sum()
    return {'缺失值数量：': s, '占比': f'{s/len(series):.5f}'}

def plot_data(data, output):
    fig, ax = plt.subplots(2, 1)
    zn = data['ZN']
    sns.histplot(zn, ax=ax[0], bins=40)
    sns.kdeplot(zn, ax=ax[1])
    fig.savefig(output / 'ZN频数统计.png', dpi=300)
    plt.close(fig)

def main():
    B_F_Data = ROOT / "data" / "Boston Housing Data.csv"
    output_dir = ROOT / 'reports' / 'exercises'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(B_F_Data)

    print('总览：\n', data,
          '\n列数值类型：\n', data.dtypes,
          '\n缺失值统计：\n', data.apply(na_count),
          '\n含缺失值行数：\n', data.isna().any(axis='columns').sum(),
          '\n数据统计：\n', data.describe().round(5),
          '\nZN列计数：\n', data['ZN'].value_counts(dropna=False),
          '\n', len(data),
          '\n', data['ZN'].count(),
          '\nRM大于6的观测：\n', data.loc[data['RM'] > 6],
          '\n按CHAS分组的MEDV均值：\n', data['MEDV'].groupby(data['CHAS']).mean())

    plot_data(data, output_dir)

if __name__ == "__main__":
    main()
