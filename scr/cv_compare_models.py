from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from train import load_data, split_data
import numpy as np
import pandas as pd
from pathlib import Path

def cv_dum(X, y, split):
    dcf = DummyClassifier(strategy='most_frequent')
    scoring = {
        'Accuracy': 'accuracy',
        'Macro-F1': make_scorer(f1_score, average='macro', labels=[0,1,2]),
    }
    cv_result = cross_validate(dcf, X, y, cv=split, scoring=scoring, n_jobs=1, return_train_score=False,
                               return_estimator=True, error_score='raise')
    return cv_result

def cv_log(X, y, split):
    log = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    scoring = {
        'Accuracy': 'accuracy',
        'Macro-F1': make_scorer(f1_score, average='macro', labels=[0, 1, 2]),
    }
    cv_result = cross_validate(log, X, y, cv=split, scoring=scoring, n_jobs=1, return_train_score=False,
                               return_estimator=True, error_score='raise')
    return cv_result

def cv_tree_3(X, y, split):
    dtcf = DecisionTreeClassifier(max_depth=3, random_state=2026)
    scoring = {
        'Accuracy': 'accuracy',
        'Macro-F1': make_scorer(f1_score, average='macro', labels=[0, 1, 2]),
    }
    cv_result = cross_validate(dtcf, X, y, cv=split, scoring=scoring, n_jobs=1, return_train_score=False,
                               return_estimator=True, error_score='raise')
    return cv_result

def cv_tree_unre(X, y, split):
    dtcf = DecisionTreeClassifier(max_depth=None, random_state=2026)
    scoring = {
        'Accuracy': 'accuracy',
        'Macro-F1': make_scorer(f1_score, average='macro', labels=[0, 1, 2]),
    }
    cv_result = cross_validate(dtcf, X, y, cv=split, scoring=scoring, n_jobs=1, return_train_score=False,
                               return_estimator=True, error_score='raise')
    return cv_result

def oof_n(result, n, split, X):
    n = n-1
    m = result['estimator'][n]
    val_pos = split[n][1]
    pred = m.predict(X.iloc[val_pos])
    return pred

def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / 'reports' / 'cv_model_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X_dev, X_test, y_dev, y_test, X_fit, X_val, y_fit, y_val = split_data(data)
    assert X_dev.index.equals(y_dev.index), '特征与标签索引或顺序不一致'
    print("自变量训练特征形状：", X_dev.shape)
    print("因变量训练形状", y_dev.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    split_list = list(skf.split(X_dev, y_dev))                                  #返回的是位置索引
    for train_index, test_index in split_list:
        overlap = np.intersect1d(train_index, test_index)
        assert len(overlap) == 0, '训练折与测试折有重叠'
    expected = np.arange(len(X_dev))
    for train_index, val_index in split_list:
        positions = np.concatenate([train_index, val_index])                    #默认按行拼接数组
        assert np.array_equal(np.sort(positions), expected), '该折存在遗漏或重复'
    val_positions = np.concatenate([v for _, v in split_list])
    assert np.array_equal(np.sort(val_positions), expected), '验证位置未恰好覆盖一次'

    cv_dum_rut = cv_dum(X_dev, y_dev, split_list)
    cv_log_rut = cv_log(X_dev, y_dev, split_list)
    cv_tree_3_rut = cv_tree_3(X_dev, y_dev, split_list)
    cv_tree_unre_rut = cv_tree_unre(X_dev, y_dev, split_list)

    dum_result = pd.DataFrame({
        'model':['Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy'],
        'fold':list(np.arange(1,6)),
        'n_fit':[len(train_index) for train_index, _ in split_list],
        'n_validation':[len(test_index) for _, test_index in split_list],
        'accuracy':cv_dum_rut['test_Accuracy'],
        'macro_f1':cv_dum_rut['test_Macro-F1']
    })
    log_result = pd.DataFrame({
        'model': ['LogisticRegression', 'LogisticRegression', 'LogisticRegression', 'LogisticRegression', 'LogisticRegression'],
        'fold': list(np.arange(1, 6)),
        'n_fit': [len(train_index) for train_index, _ in split_list],
        'n_validation': [len(test_index) for _, test_index in split_list],
        'accuracy': cv_log_rut['test_Accuracy'],
        'macro_f1': cv_log_rut['test_Macro-F1']
    })
    tree3_result = pd.DataFrame({
        'model': ['Tree_depth3', 'Tree_depth3', 'Tree_depth3', 'Tree_depth3', 'Tree_depth3'],
        'fold': list(np.arange(1, 6)),
        'n_fit': [len(train_index) for train_index, _ in split_list],
        'n_validation': [len(test_index) for _, test_index in split_list],
        'accuracy': cv_tree_3_rut['test_Accuracy'],
        'macro_f1': cv_tree_3_rut['test_Macro-F1']
    })
    treeun_result = pd.DataFrame({
        'model': ['Tree_unrestricted', 'Tree_unrestricted', 'Tree_unrestricted', 'Tree_unrestricted', 'Tree_unrestricted'],
        'fold': list(np.arange(1, 6)),
        'n_fit': [len(train_index) for train_index, _ in split_list],
        'n_validation': [len(test_index) for _, test_index in split_list],
        'accuracy': cv_tree_unre_rut['test_Accuracy'],
        'macro_f1': cv_tree_unre_rut['test_Macro-F1']
    })
    result = pd.concat([dum_result, log_result, tree3_result, treeun_result], axis=0, ignore_index=True)
    result.to_csv(output_dir / 'fold_metrics.csv', index=False)
    print(result)

    metr = pd.DataFrame({
        'model':['Dummy', 'LogisticRegression', 'Tree_depth3', 'Tree_unrestricted'],
        'accuracy_mean':[cv_dum_rut['test_Accuracy'].mean(),cv_log_rut['test_Accuracy'].mean(),cv_tree_3_rut['test_Accuracy'].mean(),cv_tree_unre_rut['test_Accuracy'].mean()],
        'accuracy_std':[cv_dum_rut['test_Accuracy'].std(ddof=0),cv_log_rut['test_Accuracy'].std(ddof=0),cv_tree_3_rut['test_Accuracy'].std(ddof=0),cv_tree_unre_rut['test_Accuracy'].std(ddof=0)],
        'macro_f1_mean':[cv_dum_rut['test_Macro-F1'].mean(),cv_log_rut['test_Macro-F1'].mean(),cv_tree_3_rut['test_Macro-F1'].mean(),cv_tree_unre_rut['test_Macro-F1'].mean()],
        'macro_f1_std':[cv_dum_rut['test_Macro-F1'].std(ddof=0),cv_log_rut['test_Macro-F1'].std(ddof=0),cv_tree_3_rut['test_Macro-F1'].std(ddof=0),cv_tree_unre_rut['test_Macro-F1'].std(ddof=0)],
    })
    metr.to_csv(output_dir / 'summary.csv', index=False)
    print('\n', metr)

    of_pre_1 = pd.DataFrame({
        'sample_index':X_dev.index[split_list[0][1]],
        'fold':1,
        'y_true':list(y_dev.iloc[split_list[0][1]]),
        'Dummy_pred':oof_n(cv_dum_rut, 1, split_list, X_dev),
        'Logistic_pred':oof_n(cv_log_rut, 1, split_list, X_dev),
        'Tree_depth3_pred':oof_n(cv_tree_3_rut, 1, split_list, X_dev),
        'Tree_unrestricted_pred':oof_n(cv_tree_unre_rut, 1, split_list, X_dev),
    })

    of_pre_2 = pd.DataFrame({
        'sample_index':X_dev.index[split_list[1][1]],
        'fold':2,
        'y_true':list(y_dev.iloc[split_list[1][1]]),
        'Dummy_pred':oof_n(cv_dum_rut, 2, split_list, X_dev),
        'Logistic_pred':oof_n(cv_log_rut, 2, split_list, X_dev),
        'Tree_depth3_pred':oof_n(cv_tree_3_rut, 2, split_list, X_dev),
        'Tree_unrestricted_pred':oof_n(cv_tree_unre_rut, 2, split_list, X_dev),
    })
    of_pre_3 = pd.DataFrame({
        'sample_index':X_dev.index[split_list[2][1]],
        'fold':3,
        'y_true':list(y_dev.iloc[split_list[2][1]]),
        'Dummy_pred':oof_n(cv_dum_rut, 3, split_list, X_dev),
        'Logistic_pred':oof_n(cv_log_rut, 3, split_list, X_dev),
        'Tree_depth3_pred':oof_n(cv_tree_3_rut, 3, split_list, X_dev),
        'Tree_unrestricted_pred':oof_n(cv_tree_unre_rut, 3, split_list, X_dev),
    })
    of_pre_4 = pd.DataFrame({
        'sample_index':X_dev.index[split_list[3][1]],
        'fold':4,
        'y_true':list(y_dev.iloc[split_list[3][1]]),
        'Dummy_pred':oof_n(cv_dum_rut, 4, split_list, X_dev),
        'Logistic_pred':oof_n(cv_log_rut, 4, split_list, X_dev),
        'Tree_depth3_pred':oof_n(cv_tree_3_rut, 4, split_list, X_dev),
        'Tree_unrestricted_pred':oof_n(cv_tree_unre_rut, 4, split_list, X_dev),
    })
    of_pre_5 = pd.DataFrame({
        'sample_index':X_dev.index[split_list[4][1]],
        'fold':5,
        'y_true':list(y_dev.iloc[split_list[4][1]]),
        'Dummy_pred':oof_n(cv_dum_rut, 5, split_list, X_dev),
        'Logistic_pred':oof_n(cv_log_rut, 5, split_list, X_dev),
        'Tree_depth3_pred':oof_n(cv_tree_3_rut, 5, split_list, X_dev),
        'Tree_unrestricted_pred':oof_n(cv_tree_unre_rut, 5, split_list, X_dev),
    })
    of_pre = pd.concat([of_pre_1, of_pre_2, of_pre_3, of_pre_4, of_pre_5], axis=0, ignore_index=True)
    of_pre = of_pre.set_index('sample_index').loc[X_dev.index].rename_axis('sample_index').reset_index()
    of_pre.to_csv(output_dir / 'oof_predictions.csv', index=False)
    print('\n', of_pre)

    cm = confusion_matrix(of_pre['y_true'], of_pre['Logistic_pred'], labels=[0, 1, 2])
    print('\n', cm)



if __name__ == "__main__":
    main()