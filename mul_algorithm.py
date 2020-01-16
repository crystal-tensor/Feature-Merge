import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion


df1 = pd.read_csv("data/feature2.txt", sep='\t')
# csv_file = 'data/feature.txt'
# ff = open(csv_file, 'r', encoding=u'utf-8', errors='ignore')
# df1 = pd.read_csv(ff)
#data_train.head(5)
frames = [df1]
#frames = [df1]
data_train = pd.concat(frames)
data_train.info()
del df1


bb=data_train.iloc[:, 5:63]
#dd=data_train.iloc[:, 3:4]
xx = bb.apply(lambda x: x.fillna(x.mean()), axis=0)
yy = data_train.iloc[:, 4:5]
# #
del bb

# Some useful parameters which will come in handy later on
titanic_train_data_X = xx
titanic_train_data_Y = yy


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    #randomforest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Feeatures from RF Classifier')
    print(str(features_top_n_rf[:10]))

    #AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    #ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))


    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # XGBClassifier
    XGB_est = XGBClassifier(random_state=0)
    XGB_param_grid = {'n_estimators': [60], 'max_depth': [9]}
    XGB_grid = model_selection.GridSearchCV(XGB_est, XGB_param_grid, n_jobs=25, cv=10, verbose=1)
    XGB_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Bset XGB Params:' + str(XGB_grid.best_params_))
    print('Top N Features Best XGB Score:' + str(XGB_grid.best_score_))
    print('Top N Features XGB Train Score:' + str(XGB_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_XGB = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': XGB_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_XGB = feature_imp_sorted_XGB.head(top_n_features)['feature']
    print('Sample 10 Features from XGB Classifier:')
    print(str(features_top_n_XGB[:10]))

    #merge the 6 models特征融合
    features_top_n = pd.concat(
         [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt,features_top_n_XGB],
         ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                       feature_imp_sorted_gb, feature_imp_sorted_dt,features_top_n_XGB], ignore_index=True)
    # features_top_n = pd.concat([features_top_n_dt],ignore_index=True).drop_duplicates()
    # features_importance = pd.concat([feature_imp_sorted_dt], ignore_index=True)
    # features_top_n = FeatureUnion([('randomforest',RandomForestClassifier()), ('AdaBoost',AdaBoostClassifier()),('ExtraTree',ExtraTreesClassifier()),
    #                                ('GradientBoosting',GradientBoostingClassifier()),('DecisionTree',DecisionTreeClassifier()),('XGBClassifier',XGBClassifier())])
    # features_importance = FeatureUnion([('randomforest',RandomForestClassifier()), ('AdaBoost',AdaBoostClassifier()),('ExtraTree',ExtraTreesClassifier()),
    #                                ('GradientBoosting',GradientBoostingClassifier()),('DecisionTree',DecisionTreeClassifier()),('XGBClassifier',XGBClassifier())])

    return features_top_n, features_importance

#特征融合
feature_to_pick = 42
feature_top_n,feature_importance = get_top_n_features(titanic_train_data_X,titanic_train_data_Y,feature_to_pick)
titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
#titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
print(feature_importance)
# File = open("data/FeatureMerge.txt", "w",encoding=u'utf-8', errors='ignore')
# for step in range(len(feature_top_n)):
#     File.write(str(feature_top_n[step]) + "\n")
File = open("data/FeatureMerge.txt", "w", encoding=u'utf-8', errors='ignore')
for step in range(len(feature_top_n)):
    File.write(str(feature_top_n[step]) + ",")
