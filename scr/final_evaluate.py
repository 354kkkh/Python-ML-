from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from train import load_data,split_data

def log_ppl(X_text, y_text, X_dev, y_dev):
    logistic = Pipeline([
        ('scaler', StandardScaler(),),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    logistic.fit(X_text, y_text)
    pre = logistic.predict(X_dev)
    pre_pro = logistic.predict_proba(X_dev)
    acc = accuracy_score(y_dev, pre)
    f1 = f1_score(y_dev, pre, average='macro')
    cm = confusion_matrix(y_dev, pre)
    return pre, pre_pro, acc, f1, cm

def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / 'reports' / 'final_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X_dev, X_test, y_dev, y_test, X_fit, X_val, y_fit, y_val = split_data(data)
    assert X_dev.index.equals(y_dev.index), '特征与标签索引或顺序不一致'
    print("自变量训练特征形状：", X_dev.shape)
    print("因变量训练形状", y_dev.shape)

    log_pre, _, log_acc, log_f1, log_cm = log_ppl(X_dev, y_dev, X_test, y_test)
    metr = pd.DataFrame([{
        'model':'LogisticRegression',
        'split':'final_test',
        'n_samples':X_test.shape[0],
        'accuracy':log_acc,
        'macro_f1':log_f1,
    }])
    metr.to_csv(output_dir / 'metrics.csv', index=False)
    print(metr)

    test_pre = pd.DataFrame({
        'sample_index':y_test.index,
        'y_true':y_test,
        'y_pred':log_pre,
    })
    test_pre.to_csv(output_dir / 'test_predictions.csv', index=False)
    print(test_pre)


if __name__ == "__main__":
    main()