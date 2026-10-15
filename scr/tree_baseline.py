from train import load_data, split_data
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score
from pathlib import Path
import pandas as pd


def classify(X_fit, y_fit, X_val, y_val):
    clf = DecisionTreeClassifier(max_depth=3, random_state=2026)
    clf.fit(X_fit, y_fit)
    pred_val = clf.predict(X_val)
    accuracy = clf.score(X_val, y_val)
    macro_f1 = f1_score(y_val, pred_val, average='macro', labels=clf.classes_)
    # 返回已拟合模型，后续评价拟合集、导出规则时无需重新训练。
    return clf, pred_val, accuracy, macro_f1


def main():
    ROOT = Path(__file__).resolve().parents[1]
    output_dir = ROOT / 'reports' / 'decision_tree'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X_dev, X_test, y_dev, y_test, X_fit, X_val, y_fit, y_val = split_data(data)
    print('开发集形状：', X_dev.shape)
    print('最终测试集形状（保留，不评价）：', X_test.shape)
    print('拟合集形状：', X_fit.shape)
    print('验证集形状：', X_val.shape)

    clf, pred_val, accuracy_val, macro_f1_val = classify(X_fit, y_fit, X_val, y_val)
    pred_fit = clf.predict(X_fit)
    metrics = pd.DataFrame([
        {
            'model': 'DecisionTree_wine',
            'max_depth': clf.max_depth,
            'split': 'fit',
            'n_samples': len(y_fit),
            'accuracy': clf.score(X_fit, y_fit),
            'macro_f1': f1_score(y_fit, pred_fit, average='macro', labels=clf.classes_),
        },
        {
            'model': 'DecisionTree_wine',
            'max_depth': clf.max_depth,
            'split': 'validation',
            'n_samples': len(y_val),
            'accuracy': accuracy_val,
            'macro_f1': macro_f1_val,
        },
    ])
    metrics.to_csv(output_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
    print('\n训练与验证指标：\n', metrics.to_string(index=False))
    print('\n验证集预测：', pred_val)

    rules = export_text(clf, feature_names=X_fit.columns.tolist(), decimals=6)
    (output_dir / 'tree_rules.txt').write_text(rules, encoding='utf-8')
    print('\n树规则：\n', rules)
    print('实际树深度：', clf.get_depth(), '；叶节点数：', clf.get_n_leaves())

    # 用同一个拟合样本对照上面的规则，手工核对分类路径。
    sample = X_fit.iloc[[0]]
    sample_index = sample.index[0]
    print('\n手工核对样本的原索引：', sample_index)
    print('样本特征：\n', sample.T)
    print('真实类别：', y_fit.loc[sample_index], '；模型预测：', clf.predict(sample)[0])
    print('\n指标和树规则已保存至：', output_dir)

if __name__ == '__main__':
    main()
