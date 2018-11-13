import pandas as pd
import gc
import numpy as np
from pandas import Series, DataFrame
from sklearn  import preprocessing
from sklearn import model_selection
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector

aa = pd.read_csv("data/train_1.txt", sep='\t')
#bb = pd.read_csv("data/feature_top_decisiontree_ExtraTree_list.txt", sep='\t')
#x_data = preprocessing.minmax_scale(aa.iloc[:, 4:6759].values, feature_range=(-1,1))
# alg1 = RandomForestClassifier(random_state=1, n_estimators=62, min_samples_split=2, min_samples_leaf=1)
# predictors = adalist
# pipe1 = make_pipeline(ColumnSelector(cols=x_data[:, 0:20]), alg1)
data_train = aa.dropna(axis = 0)
print(data_train)