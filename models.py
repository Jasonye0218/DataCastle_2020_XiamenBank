import pandas as pd
import numpy as np
import warnings

import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def eval_error(pred, train_set):
    labels = train_set.get_label()
    pred = pred.reshape((3, int(len(pred) / 3))).T
    y_pred = pred.argmax(axis=1)
    score = cohen_kappa_score(labels, y_pred)
    return 'kappa_score', score, True


def xgb_eval_error(pred, train_set):
    labels = train_set.get_label()
    y_pred = pred.argmax(axis=1)  # xgb直接生成矩阵形式
    score = cohen_kappa_score(labels, y_pred)
    return '1_sub_kappa_score', 1 - score


def lgb_model_train(df, trainlabel, cate_cols, test_, feature, num_class):
    '''
    @param df: 训练数据 DataFrame
    @param trainlabel：训练标签 string  eg. 'label'
    @param cate_cols: 类别变量名 list  eg. ['col1','col2'...]
    @param test_ : 测试数据 DataFrame
    @param feature ：所有训练特征 list  eg. ['feat1','feat2'...]

    @return sub_preds: 预测数据

    '''

    train_ = df.copy()

    n_splits = 5
    oof_lgb = np.zeros([len(train_), num_class])
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    stratifiedKfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    sub_preds = np.zeros([test_.shape[0], num_class])
    sub_preds1 = np.zeros([test_.shape[0], n_splits])
    use_cart = True
    cate_cols = cate_cols
    label = trainlabel
    pred = list(feature)


    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        # 'metric':'multi-error',
        'num_class': num_class,
        'num_leaves': 60,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 5,
        'min_data_in_leaf': 20,
        'max_depth': -1,
        'nthread': 8,
        'verbose': 1,
        #'is_unbalanace':True,
        #'lambda_l1': 0.4,
        #'lambda_l2': 0.5,
        # 'device': 'gpu'
    }

    valid_kappa = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_[pred], train_[[label]]), start=1):
        print('the %s training start ...' % n_fold)

        train_x, train_y, train_weight = train_[pred].iloc[train_idx], train_[[label]].iloc[train_idx], \
                                         train_[['weight']].iloc[train_idx]
        valid_x, valid_y, valid_weight = train_[pred].iloc[valid_idx], train_[[label]].iloc[valid_idx], \
                                         train_[['weight']].iloc[valid_idx]

        print(train_y.shape)

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols,
                                 weight=train_weight.values.flatten(order='F'))
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=eval_error,
        )

        sub_preds += clf.predict(test_[pred].values, num_iteration=clf.best_iteration) / folds.n_splits
        # sub_preds1[:, n_fold - 1] = clf.predict(test_[pred].values, num_iteration=400).argmax(axis=1)
        valid_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)
        oof_lgb[valid_idx] = valid_pred
        y_pred = valid_pred.argmax(axis=1)
        fold_valid_kappa = cohen_kappa_score(valid_y, y_pred)
        valid_kappa += fold_valid_kappa

    print('The average valid kappa_score:{:.6f}'.format(valid_kappa / n_splits))

    return sub_preds, oof_lgb, clf, sub_preds1


def xgb_model_train(df, trainlabel, cate_cols, test_, feature, num_class):
    '''
    @param df: 训练数据 DataFrame
    @param trainlabel：训练标签 string  eg. 'label'
    @param cate_cols: 类别变量名 list  eg. ['col1','col2'...]
    @param test_ : 测试数据 DataFrame
    @param feature ：所有训练特征 list  eg. ['feat1','feat2'...]

    @return sub_preds: 预测数据

    '''
    train_ = df.copy()
    auc = []
    n_splits = 5
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    stratifiedKfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    sub_preds = np.zeros([test_.shape[0], num_class])
    sub_preds1 = np.zeros([test_.shape[0], n_splits])
    cate_cols = cate_cols
    label = trainlabel
    pred = list(feature)

    # train_['L'] = 1
    # test_['L'] = 0
    # data1 = pd.concat([train_, test_])
    # for item in cate_cols:
    #     item_dummies = pd.get_dummies(data1[item])
    #     item_dummies.columns = [item + str(i + 1) for i in range(item_dummies.shape[1])]
    #     data = pd.concat([data1, item_dummies], axis=1)
    # data.drop(cate_cols, axis=1, inplace=True)

    params = {
        'booster': 'gbtree',  # 提升类型
        'objective': 'multi:softprob',  # 目标函数
        'num_class': num_class,
        # 'eval_metric': 'auc',  # 评价函数
        'eta': 0.07,  # 学习率 ，一般0.0几
        'gamma': 0.1,
        'max_depth': 8,
        'alpha': 0,
        'lambda': 0,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': 3,
        'silent': 0,
        'nthread': -1,
        'missing': 1,
        'seed': 2019,
    }

    valid_kappa = 0
    feature_importance_df = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_[pred], train_[[label]]), start=1):
        print('the %s training start ...' % n_fold)

        train_x, train_y, train_weight = train_[pred].iloc[train_idx], train_[[label]].iloc[train_idx], \
                                         train_[['weight']].iloc[train_idx]
        valid_x, valid_y, valid_weight = train_[pred].iloc[valid_idx], train_[[label]].iloc[valid_idx], \
                                         train_[['weight']].iloc[valid_idx]

        print(train_y.shape)
        dtrain = xgb.DMatrix(train_x, train_y, weight=train_weight.values.flatten(order='F'))
        dvalid = xgb.DMatrix(valid_x, valid_y)
        watchlist = [(dvalid, 'eval')]

        clf = xgb.train(params=params, dtrain=dtrain, num_boost_round=2000, evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=100, feval=xgb_eval_error)

        dtest = xgb.DMatrix(test_[pred])
        sub_preds += clf.predict(dtest, ntree_limit=clf.best_iteration) / folds.n_splits
        valid_pred = clf.predict(dvalid, ntree_limit=clf.best_iteration)
        y_pred = valid_pred.argmax(axis=1)
        fold_valid_kappa = cohen_kappa_score(valid_y, y_pred)
        valid_kappa += fold_valid_kappa

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = clf.get_fscore().keys()
        fold_importance_df["importance"] = clf.get_fscore().values()
        fold_importance_df["fold"] = n_fold
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('The average valid kappa_score:{:.6f}'.format(valid_kappa / n_splits))

    ## plot feature importance
    cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
                                                                                                   ascending=False).index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',
                                                                                                    ascending=False)
    plt.figure(figsize=(8, 15))
    sns.barplot(y="Feature",
                x="importance",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()

    return sub_preds, clf, sub_preds1


def cat_model_train(df, trainlabel, cate_cols, test_, feature, num_class):
    '''
    @param df: 训练数据 DataFrame
    @param trainlabel：训练标签 string  eg. 'label'
    @param cate_cols: 类别变量名 list  eg. ['col1','col2'...]
    @param test_ : 测试数据 DataFrame
    @param feature ：所有训练特征 list  eg. ['feat1','feat2'...]

    @return sub_preds: 预测数据

    '''
    train_ = df.copy()
    n_splits = 5
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    stratifiedKfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    cate_cols = cate_cols
    label = trainlabel
    pred = list(feature)

    for index, value in enumerate(train_[pred].columns):
        train_[value].fillna(0, inplace=True)
        test_[value].fillna(0, inplace=True)

    oof = np.zeros([len(train_[pred]), 7])
    predictions = np.zeros(test_.shape[0])

    for fold_, (train_index, test_index) in enumerate(folds.split(train_[pred], train_[[label]])):
        print("fold n°{}".format(fold_ + 1))

        train_x, train_y, train_weight = train_[pred].iloc[train_index], train_[[label]].iloc[train_index], \
                                         train_[['weight']].iloc[train_index]
        test_x, test_y, test_weight = train_[pred].iloc[test_index], train_[[label]].iloc[test_index], \
                                      train_[['weight']].iloc[test_index]

        cbt_model = CatBoostRegressor(iterations=5000, learning_rate=0.05, max_depth=7, verbose=100,
                                      early_stopping_rounds=50, eval_metric='MAE', cat_features=cate_cols,
                                      # task_type = 'GPU'
                                      )
        cbt_model.fit(train_x, train_y, eval_set=(test_x, test_y))
        predictions += cbt_model.predict(test_[pred]) / 5

    # 阈值卡比例
    predictions.tolist()
    res = [0 for i in range(len(predictions))]
    sum1 = pd.read_csv('./sum1.csv')
    sum1 = sum1[['cust_no']]

    tem = sorted(predictions)
    yuzhi1 = tem[11738]
    yuzhi2 = tem[76722 - 49011]
    for i in range(len(res)):
        if predictions[i] < yuzhi1:
            res[i] = -1
        elif predictions[i] < yuzhi2:
            res[i] = 0
        else:
            res[i] = 1

    test_['label'] = res
    # result['label']=result['label'].astype("int64")
    test_ = test_[['cust_no', 'label']]
    result = pd.merge(sum1, test_, on='cust_no', how='left')

    return predictions, result


def main():

    Train_data = pd.read_csv('./train_data.csv')
    cust_avli_test = pd.read_csv('./test_data.csv')
    Train_data['weight'] = Train_data.label.map({1: 1.03, 2: 0.58, 0: 1})
    # feature = Train_data.drop(['cust_no', 'label', 'weight'], axis=1).columns
    cate_cols = ['I3', 'I5', 'I8', 'I10', 'I12', 'I13', 'I14']

    cols = np.load('./test/importance_cols.npy', allow_pickle=True)
    cols.tolist()
    select_features = cols[0:300]

    cate_cols = [x for x in cate_cols if x in select_features] # 求筛选的特征与cate特征的交集

    # 调用xgb
    sub_preds, clf, sub_preds1 = xgb_model_train(Train_data, trainlabel='label', cate_cols=cate_cols,
                                                      test_=cust_avli_test, feature=select_features, num_class=3)

    # 调用lgb
    sub_preds, oof_lgb, clf, sub_preds1 = lgb_model_train(Train_data, trainlabel='label', cate_cols=cate_cols,
                                                      test_=cust_avli_test, feature=select_features, num_class=3)

    cust_avli_test['label'] = sub_preds.argmax(axis=1) - 1  # 原标签+1了
    cust_avli_test[['cust_no', 'label']].to_csv('./xgb_all_in.csv', index=None)

    # 调用cat回归
    predictions, result = cat_model_train(Train_data, trainlabel='label', cate_cols=cate_cols,
                                                      test_=cust_avli_test, feature=select_features, num_class=3)
    result.to_csv('./test/cat_all_in.csv', index=None)


if __name__ == "__main__":
    main()
