from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from train import load_data, split_data

def cv(X, y, split):
    logistic = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    scoring = {
        'accuracy': 'accuracy',
        'macro_f1': make_scorer(f1_score, average='macro', labels=[0, 1, 2]),
    }
    cv_results = cross_validate(logistic, X, y, cv=split, n_jobs=1, scoring=scoring, return_train_score=False, error_score='raise')
    return cv_results

def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / 'reports' / 'cross_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X_dev, X_test, y_dev, y_test, X_fit, X_val, y_fit, y_val = split_data(data)
    print("自变量训练特征形状：", X_dev.shape)
    print("因变量训练形状", y_dev.shape)
    print("测试特征形状：", X_test.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    Index = list(skf.split(X_dev, y_dev))
    for train_idx, val_idx in Index:
        overlap = np.intersect1d(train_idx, val_idx)
        assert len(overlap) == 0, '训练折和测试折存在重叠索引'
    val_positions = np.concatenate([v for _, v in Index])
    print(len(val_positions))
    print(Index)

    cv_rst = cv(X_dev, y_dev, Index)
    print('五折的 Accuracy:', cv_rst['test_accuracy'], '\n五折的 Macro-F1:', cv_rst['test_macro_f1'])

    fold_metrics = pd.DataFrame({
        'fold': list(range(1, len(Index) + 1)),
        'n_fit': [len(train_idx) for train_idx, _ in Index],
        'n_validation': [len(val_idx) for _, val_idx in Index],
        'accuracy': cv_rst['test_accuracy'],
        'macro_f1': cv_rst['test_macro_f1'],
    })
    fold_metrics.to_csv(output_dir / 'fold_metrics.csv', index=False)
    print(fold_metrics)

    print('Accuracy五折均值：', format(cv_rst['test_accuracy'].mean(), '.4f'), '\nAccuracy五折标准差：', format(cv_rst['test_accuracy'].std(), '.4f'))
    print('Macro-F1五折均值:', format(cv_rst['test_macro_f1'].mean(), '.4f'), '\nMacro-F1五折标准差：', format(cv_rst['test_macro_f1'].std(), '.4f'))

if __name__ == '__main__':
    main()
