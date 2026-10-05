from train import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dummy_predict(x_fit, y_fit, x_test, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(x_fit, y_fit)                           #拟合模型
    predic_dum =  dummy_clf.predict(x_test)
    predic_dum_sco = dummy_clf.score(x_test, y_test)
    macro_f1 = f1_score(y_test, predic_dum, average='macro')
    cm = confusion_matrix(y_test, predic_dum, labels=[0,1,2])
    return predic_dum, predic_dum_sco, macro_f1, cm

def logistic_predict(x_fit, y_fit, x_test, y_test):
    logistic = Pipeline([
        ('scaler', StandardScaler()),                    # 对以后输入的自变量自动标准化
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    logistic.fit(x_fit, y_fit)                  #拟合模型,自动保存到pipeline中
    pre_log = logistic.predict(x_test)
    pre_log_pro = logistic.predict_proba(x_test)
    pre_log_sco = logistic.score(x_test, y_test)
    macro_f1 = f1_score(y_test, pre_log, average='macro')
    cm = confusion_matrix(y_test, pre_log, labels=[0,1,2])
    return pre_log, pre_log_pro, pre_log_sco, macro_f1, cm

def plot_cm(cm, path, str):         #混淆矩阵可视化
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(path / str)
    plt.close()

def main():
    out = ROOT / 'reports' / 'baseline'
    out.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, Y_train, Y_test, x_small_fit, x_small_test, y_small_fit, y_small_test = split_data(df)

    predic_dum, predic_dum_sco, macro_f1_dum, cm_dum = dummy_predict(x_small_fit, y_small_fit, x_small_test, y_small_test)
    print('预测结果：', predic_dum, '\n预测准确率：', predic_dum_sco,
          '\n宏平均：', macro_f1_dum, '\n混淆矩阵：\n', cm_dum)

    pre_log, pre_log_pro, pre_log_sco, macro_f1_log, cm_log = logistic_predict(x_small_fit, y_small_fit, x_small_test, y_small_test)
    print('\n逻辑回归预测结果：', pre_log, '\n逻辑回归所得概率：', pre_log_pro,
          '\n逻辑回归预测准确率：', pre_log_sco, '\n逻辑回归宏平均：', macro_f1_log, '\n混淆矩阵：\n', cm_log)

    plot_cm(cm_dum, out, 'cm_dum.png')           #保存混淆矩阵图片
    plot_cm(cm_log, out, 'cm_log.png')

#导出模型结果的操作顺序：先构造DataFrame，再用.to_csv方法导出表格
    n_val = len(y_small_test)
    # 检查特征、标签和预测是否对应同一批样本
    assert x_small_test.index.equals(y_small_test.index)
    assert len(predic_dum) == len(pre_log) == n_val

    # 每个字典代表一个模型的一行记录
    metrics = pd.DataFrame([
        {"model": "Dummy",
         "n_validation": n_val,
         "accuracy": predic_dum_sco,
         "macro_f1": macro_f1_dum,
        },
        {"model": "LogisticRegression",
         "n_validation": n_val,
         "accuracy": pre_log_sco,
         "macro_f1": macro_f1_log,
        },
    ])

    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    # 字典的键是列名；每列放入按相同顺序排列的数据
    predictions = pd.DataFrame({
        "sample_index": x_small_test.index.to_numpy(),
        "y_true": y_small_test.to_numpy(),
        "dummy_pred": predic_dum,
        "logistic_pred": pre_log,
    })

    # 每条逻辑回归预测是否正确
    predictions["logistic_correct"] = np.where(predictions["y_true"] == predictions["logistic_pred"], True, False)

    predictions.to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n指标表：\n", metrics)
    print("\n验证预测前5行：\n", predictions.head())

if __name__ == '__main__':
    main()