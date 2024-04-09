import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import preprocessing


  # Import dataset

  tmp_sql = "select * from temp_model.model_PF_total a inner join temp_model.model_all_channel_time_total b on a.customer_no =   b.customer_no"
  print(tmp_sql)
  df = rms_utils.get_pd_by_jbeapi(tmp_sql)
  df = df.drop(['Customer_NO'], axis=1)
  df.head(10)
  df.shape
  print(df.dtypes)
  print(df["line_time_flag"].unique())
  df_1 = df
  df_1.head(10)
  df_1.shape
  print(df_1.dtypes)
  print(df_1["line_time_flag"].unique())
  line_timing_model = spark.createDataFrame(df)
  line_timing_model.createOrReplaceTempView("line_timing_model")
  line_timing_model.write.parquet("/user/line_time_model/line_timing_model",compression="gzip",mode ='overwrite')
  
  # Data Preprocessing 
    
  cols = df.select_dtypes(object).columns
  ## remove all whitespace
  df[cols] = df[cols].apply(lambda x: x.str.strip())
  print(df["line_time_flag"].unique())
  print(df["membership"].unique())
  print(df["loy_flag"].unique())
  print(df["mbb_time_flag"].unique())
  print(df["mbb_cnt_flag"].unique())
  print(df["mbb_recen_flag"].unique())
  print(df["line_no_1_flag"].unique())
  
  ## handle missing value
  df.replace('NA','0',inplace=True)
  df.fillna(0,inplace=True)
  ## transfer data type to float
  df = df.astype(float)
  print(df.dtypes)
  df.head(10)

  # Feature Engineering
  ## define Y for training (model output)
  df[df["line_time_flag"] == 2].head(10)
  df_pos = df[df["line_time_flag"] == 2]
  df_pos.head(10)
  df_pos.shape
  pose_count = df_pos.shape
  pose_count

  ## define X1 for training (never click one online ads before) 2/3
  neg_count = int(pose_count[0]*(2/3))
  neg_count
  df_neg = df[df["line_time_flag"] == 5]
  df_neg = df_neg[:neg_count]
  df_neg.shape

  ## define X2 for training (never click on online ads during specific time periods) 1/3
  neg_count_1 = int(pose_count[0] - neg_count)
  neg_count_1
  df_no2 = df[df["line_no_2_flag"] == 1]
  df_neg2 = df_no2[:neg_count_1]
  df_neg2.shape
  df_neg2.head(10)

  ## concat X1 and X2
  df_neg_all = pd.concat([df_neg,df_neg2])
  print(df_neg_all["line_time_flag"].unique())
  df_neg_all.shape
  df_neg_all.head()
  
  ## concat X and Y (training data)
  df_pos.shape
  df_pos.loc[df_pos["line_time_flag"] == 2,"line_time_flag"] = 1
  df_pos.head()
  print(df_pos["line_time_flag"].unique())
  df_neg_all.loc[df_neg_all["line_time_flag"].isin([1,3,5]),"line_time_flag"] = 0
  df_neg_all.head(10)
  print(df_neg_all["line_time_flag"].unique())
  df_neg_all = shuffle(df_neg_all)
  all_df = pd.concat([df_pos,df_neg_all])
  all_df = shuffle(all_df)
  all_df.info()
  all_df.head(10)
  cols = all_df.columns
  features = cols.drop(["line_time_flag","line_no_1_flag","line_no_2_flag","line_no_3_flag"])
  lables = "line_time_flag"

  # Apply XGBoost for model training
  X = all_df[features].values
  y = all_df[lables].values
  ##scale features to a specified range
  scaler = MinMaxScaler()
  X_nor = scaler.fit_transform(X)
  X_nor = scaler.transform(X)
  train_X, test_X, train_y, test_y = train_test_split(X_nor, y, test_size=0.3, random_state=1)
  train_X.shape
  TEXT_MODEL_TUNED_PARAMETERS = {
                      'tree_method': ['hist'],
                      #'tree_method': ['gpu_hist'],
                      #'gpu_id': [0],
                      'n_estimators': [200] ,
                      'max_depth': [10],
                      'eta': [0.01],
                      'seed': [0],
                      'objective': ['binary:logistic'],
                      'colsample_bytree': [0.8],
                      'subsample': [0.8],
                      'learning_rate': [0.1]}

  def fit_cv_model(method, tuned_parameters, train_X, train_y, cv_fold):
      scoring_list = ['accuracy', 'precision']
      cv = GridSearchCV(method(), tuned_parameters, cv=cv_fold,
      scoring=scoring_list, refit='accuracy', return_train_score=True)
      cv.fit(train_X, train_y)
      best_params = cv.best_params_.copy()
      best_params.update({'tree_method': 'hist'})
      model = method(**best_params,verbosity = 0,silent=True,nthread=2)
      model.fit(train_X, train_y)
      return model, cv
      model, cv = fit_cv_model(XGBClassifier, TEXT_MODEL_TUNED_PARAMETERS, train_X, train_y, 5)
      model_fit = model.fit(train_X,train_y)
      test_y_predicted = model_fit.predict(test_X)

      ## XGBoost model result  
      accuracy = metrics.accuracy_score(test_y, test_y_predicted)
      print(accuracy)


from pickle import dump

f_id = open('/home/cdsw/scaler_02_2402.pkl', 'wb')

dump(scaler, f_id)

f_id.close()

f_id = open('/home/cdsw/model_02_2402.pkl', 'wb')

dump(model_fit, f_id)

f_id.close()

print(np.unique(test_y_predicted))

np.count_nonzero(test_y_predicted == 1)


print(model.feature_importances_)


plt.bar(range(len(model.feature_importances_)),model.feature_importances_)
plt.show()

from xgboost import plot_importance
plot_importance(model)

plt.show()

features


model.get_booster().get_score(importance_type = 'weight')

