from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from train import load_data, split_data


def evaluate(model, X, y):
    pred = model.predict(X)
    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average='macro', labels=[0, 1, 2])
    return accuracy, macro_f1


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'reports' / 'model_comparison'
    output.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X_dev, X_test, y_dev, y_test, X_fit, X_val, y_fit, y_val = split_data(data)

    assert X_fit.shape == (113, 13) and X_val.shape == (29, 13)
    assert X_test.shape == (36, 13)
    for X, y in ((X_fit, y_fit), (X_val, y_val), (X_test, y_test)):
        assert X.index.equals(y.index)
    for left, right in ((X_fit, X_val), (X_fit, X_test), (X_val, X_test)):
        assert left.index.intersection(right.index).empty

    models = {
        'Dummy': DummyClassifier(strategy='most_frequent'),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000)),
        ]),
        'Tree_depth3': DecisionTreeClassifier(max_depth=3, random_state=2026),
        'Tree_unrestricted': DecisionTreeClassifier(max_depth=None, random_state=2026),
    }

    rows = []
    for name, model in models.items():
        # 每个方案只拟合一次，复用同一个模型评价拟合集和验证集。
        model.fit(X_fit, y_fit)
        for split, X, y in (('fit', X_fit, y_fit), ('validation', X_val, y_val)):
            accuracy, macro_f1 = evaluate(model, X, y)
            rows.append({
                'model': name,
                'split': split,
                'n_samples': len(y),
                'accuracy': accuracy,
                'macro_f1': macro_f1,
            })
        if isinstance(model, DecisionTreeClassifier):
            print(f'{name}：实际深度={model.get_depth()}，叶节点数={model.get_n_leaves()}')

    df = pd.DataFrame(rows)
    df.to_csv(output / 'metrics.csv', index=False)
    print(df.to_string(index=False))

    # 复用上面已拟合的四个模型，不重新训练或划分数据。
    dummy_pred = models['Dummy'].predict(X_val)
    log_pred = models['LogisticRegression'].predict(X_val)
    tree_3_pred = models['Tree_depth3'].predict(X_val)
    tree_un_pred = models['Tree_unrestricted'].predict(X_val)
    cm_tree_3 = confusion_matrix(y_val, tree_3_pred, labels=[0, 1, 2])

    df_val_pre = pd.DataFrame({
        'sample_index': X_val.index,
        'y_true': y_val,
        'Dummy_pred': dummy_pred,
        'Logistic_pred': log_pred,
        'Tree_depth3_pred': tree_3_pred,
        'Tree_unrestricted_pred': tree_un_pred,
    })
    df_val_pre.to_csv(output / 'validation_predictions.csv', index=False)
    print(df_val_pre.to_string(index=False))
    print(cm_tree_3)


if __name__ == '__main__':
    main()
