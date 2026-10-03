import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B_F_Data = ROOT / "data" / "Boston Housing Data.csv"
data = pd.read_csv(B_F_Data)

print('列数值类型：\n',data.dtypes)
print('\n数据统计：\n',data.describe())
print('\n ZN列计数：\n',data['ZN'].value_counts())
