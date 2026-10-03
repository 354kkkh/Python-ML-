from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_wine(as_frame=True)
df = pd.concat([data['data'], data['target']], axis=1)
print('特征维度：\n', df.columns, '\n 数据类型： \n',df.dtypes, '\n 数据：\n', df)

#测试集划分
X = df.iloc[:, :13]
Y = df.iloc[:, 13]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2026
)


