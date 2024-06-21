os.chdir("/home/cdsw/")
import pandas as pd  
from  libs.dataaccess import connector
conn = connector.connector()
conn.local()
conn.set_jdbc_login("Z00043597","28585097allp@")

spdf = conn.parquet_query("/user/Z00043597/line_time_model/time_score_2402")

conn.set_jdbc_target("temp_sweep","z43597_2402_tm_scores")
conn.jdbc_overwrite(spdf)

time_score_write = spark.read.parquet("/user/Z00043597/line_time_model/time_score_2308")

























# coding: utf-8

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

###從RMS導入資料
  tmp_sql = "select * from temp_model.model_PF_total a inner join temp_model.model_all_channel_time_total b on a.customer_no = b.customer_no"
  print(tmp_sql)
  RMS_df = rms_utils.get_pd_by_jbeapi(tmp_sql)
  RMS_df.shape
  RMS_df.head()
###把table備份，以防原本df資料被弄錯
  df = RMS_df
  df1 = RMS_df
  df.head()
  df.shape
  df1.head()
  df1.shape
###基本觀察
  duplicate = df[df.duplicated()]
  duplicate
  print(df.dtypes)
  print(df["line_time_flag"].unique())
###避免欄位名重複
  col_names = df.columns.values
  col_names[0] = 'Changed'
  df.columns = col_names
  col_names
  df = df.drop(['Changed'], axis=1)
  df.head()
  df.shape

  
###df去除空白
  cols = df.select_dtypes(object).columns
  cols
  df[cols] = df[cols].apply(lambda x: x.str.strip())
  print(df["line_time_flag"].unique())
  print(df["membership"].unique())
  print(df["loy_flag"].unique())
  print(df["mbb_time_flag"].unique())
  print(df["mbb_cnt_flag"].unique())
  print(df["mbb_recen_flag"].unique())
  #df NA轉成0
  df.replace('NA','0',inplace=True)
  #df none轉成0
  df.fillna(0,inplace=True)
  #df 欄位轉換type
  df = df.astype(float)
  print(df.dtypes)
  df.head()
  
###提取特徵，拿掉客戶編號
cols = df.columns
df.shape
cols
features = cols.drop(["line_time_flag","line_no_1_flag","line_no_2_flag","line_no_3_flag"])
features = features.drop("Customer_NO")
len(np.unique(features))


###模型預測前置準備
X = df[features].values
scaler = MinMaxScaler()
from pickle import load
scaler = load(open('/home/cdsw/scaler_03_2402.pkl', 'rb'))
X_nor = scaler.transform(X)
X_nor.shape
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
                      'learning_rate': [0.1]
}


def fit_cv_model(method, tuned_parameters, cv_fold):
    scoring_list = ['accuracy', 'precision']
    cv = GridSearchCV(method(), tuned_parameters, cv=cv_fold,
                    scoring=scoring_list, refit='accuracy', return_train_score=True)
    #cv.fit(train_X, train_y)
    #best_params = cv.best_params_.copy()
    #best_params.update({'tree_method': 'gpu_hist'})
    #best_params.update({'tree_method': 'hist'})
    model = method(**tuned_parameters,verbosity = 0,silent=True,nthread=2)
    #model.fit(train_X, train_y)
    return model, cv


###模型預測
model, cv = fit_cv_model(XGBClassifier, TEXT_MODEL_TUNED_PARAMETERS, 5)
model = load(open('/home/cdsw/model_03_2402.pkl', 'rb'))
pred_all = model.predict_proba(X_nor)
pred_all_bi = model.predict(X_nor)
pred_all_bi
pred_all.shape
type(pred_all)
pred_all_list = pred_all[:,1]
len(pred_all_list)
type(pred_all_list)

cno_list = list(df["Customer_NO"].values)
cno_list[:10]
cno_df = pd.DataFrame(cno_list,columns=['cno'])
cno_df.head()
score_df = pd.DataFrame(pred_all_list,columns=['score'])
score_df.head()
u_rms_df = pd.concat([cno_df,score_df],axis=1)
u_rms_df.head()

time_score_01 = u_rms_df
time_score_01.head()

time_score_02 = u_rms_df
time_score_02.head()

time_score_03 = u_rms_df
time_score_03.head()

time_score_01.shape
time_score_02.shape
time_score_03.shape
time_score = time_score_01.merge(time_score_02,on='cno').merge(time_score_03,on='cno')
time_score.head()
time_score.shape

time_score_2402 = spark.createDataFrame(time_score)
time_score_2402.write.parquet("/user/Z00043597/line_time_model/time_score_2402",compression="gzip",mode ='overwrite')                                   



#導回RMS
spark.stop()

import pyspark.sql.window as W
import os
from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime as dt
 
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("example") \
    .config("spark.jars","/home/cdsw/terajdbc4.jar,/home/cdsw/tdgssconfig.jar") \
    .config("spark.hadoop.dfs.client.block.write.replace-datanode-on-failure.best-effort","True") \
    .config("spark.kryoserializer.buffer.max",2047) \
    .enableHiveSupport() \
    .getOrCreate()

database = 'temp_sweep'
table_chartset = 'WINDOWS-950'
username = 'Z00043597'
password = "28585097allp@"
driver = 'com.teradata.jdbc.TeraDriver'
server = '10.23.96.39'
url = 'jdbc:teradata://{}/DATABASE={},Client_Charset={}'.format(server, database, table_chartset)

time_score_write = spark.read.parquet("/user/Z00043597/line_time_model/time_score_2402")
time_score_write.cache()
time_score_write.count()
time_score_write.limit(10).toPandas()
time_score_write.write.format("jdbc").mode("append").options(url=url, driver=driver,dbtable="temp_model.time_tag_2402",user=username,password=password).save()





for loc,acc in enumerate(pred_all_bi):
    c_no = df.iloc[loc:loc+1]["Customer_NO"].values[0]
    print(c_no,acc)
    if loc>10:
        break
        
for loc,acc in enumerate(pred_all):
    c_no = df.iloc[loc:loc+1]["Customer_NO"].values[0]
    print(c_no,acc[1])
    if loc>10:
        break


accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)


# In[ ]:


from pickle import dump


# In[ ]:


f_id = open('scaler.pkl', 'wb')


# In[ ]:


dump(scaler, f_id)


# In[ ]:


f_id.close()


# In[ ]:


from pickle import load


# In[ ]:


scaler = load(open('scaler.pkl', 'rb'))


# In[ ]:


model_fit.save_model("model.txt")


# In[ ]:


import xgboost as xgb


# In[ ]:


model_load = xgb.Booster()
model_load.load_model("model.txt")


# In[ ]:


model_load.best_ntree_limit


# In[ ]:


print(np.unique(test_y_predicted))


# In[ ]:


np.count_nonzero(test_y_predicted == 1)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


plt.bar(range(len(model.feature_importances_)),model.feature_importances_)
plt.show()


# In[ ]:


from xgboost import plot_importance
plot_importance(model)

plt.show()


# In[ ]:


features


# In[ ]:


model.get_booster().get_score(importance_type = 'weight')

