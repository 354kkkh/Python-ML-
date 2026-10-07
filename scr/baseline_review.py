from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import pandas as pd
from pathlib import Path

def load_data():
    data = load_wine(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    return df

def split_data(df, y:str, others=()):
    ex_x = [y, *others]
    X = df.drop(columns=ex_x)
    Y = df[y]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2026, stratify=Y
    )

    overlap = x_train.index.intersection(y_test.index)
    assert len(overlap) == 0, '训练集和测试集存在重叠索引'
    return x_train, x_test, y_train, y_test

def dum_model(X_train, Y_train, X_test, Y_test):
    dum_clf = DummyClassifier(strategy='most_frequent')
    dum_clf.fit(X_train, Y_train)
    dum_pred = dum_clf.predict(X_test)
    dum_pred_scr = dum_clf.score(X_test, Y_test)
    macro_f1 = f1_score(Y_test, dum_pred, average='macro')
    cm = confusion_matrix(Y_test, dum_pred)
    return dum_pred, dum_pred_scr, macro_f1, cm

def log_model(X_train, Y_train, X_test, Y_test):
    L_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    L_pipeline.fit(X_train, Y_train)
    L_pred = L_pipeline.predict(X_test)
    L_pred_pro = L_pipeline.predict_proba(X_test)
    L_pred_scr = L_pipeline.score(X_test, Y_test)
    macro_f1 = f1_score(Y_test, L_pred, average='macro')
    cm = confusion_matrix(Y_test, L_pred)
    return L_pred, L_pred_pro, L_pred_scr, macro_f1, cm

def plot_cm(cm, path, name: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(path / name)
    plt.close()

def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / 'reports' / 'baseline' / 'baseline_review'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, Y_train, Y_test = split_data(df, 'target')
    print('训练集形状：', X_train.shape)
    print('测试集形状：', X_test.shape)
    print('训练集各类别个数：\n', Y_train.value_counts())
    print('测试集各类别个数：\n', Y_test.value_counts())

    df_vfit = pd.concat([X_train, Y_train], axis=1)
    x_train, x_test, y_train, y_test = split_data(df_vfit, 'target')
    print('\n训练集中用于拟合数据的形状：', x_train.shape)
    print('训练集中用于验证数据的形状：', x_test.shape)
    print('训练集中用于拟合数据的各类别数：\n', y_train.value_counts())
    print('训练集中用于验证数据的各类别数：\n', y_test.value_counts())

    Dum_pred, Dum_pred_scr, Dum_macro_f1, Dum_cm = dum_model(x_train, y_train, x_test, y_test)
    print('\n预测结果：', Dum_pred, '\n预测准确率：', Dum_pred_scr,
          '\n宏平均：', Dum_macro_f1, '\n混淆矩阵：\n', Dum_cm)
    L_pred, L_pred_pro, L_pred_scr, L_macro_f1, L_cm = log_model(x_train, y_train, x_test, y_test)
    print('\n逻辑回归预测结果：', L_pred, '\n逻辑回归所得概率：\n', L_pred_pro,
          '\n逻辑回归预测准确率：', L_pred_scr, '\n逻辑回归宏平均：', L_macro_f1, '\n混淆矩阵：\n', L_cm)

    plot_cm(Dum_cm,output_dir,'cm_dum.png')
    plot_cm(L_cm,output_dir,'cm_log.png')
    print('混淆矩阵已保存至reports文件夹下')

if __name__ == '__main__':
    main()

'''

'''