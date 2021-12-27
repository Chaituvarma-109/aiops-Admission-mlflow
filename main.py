import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


adm_df = pd.read_csv("Admission_Prediction.csv")

adm_df.drop(columns='Serial No.', inplace=True)

adm_df['GRE Score'] = adm_df['GRE Score'].fillna(adm_df['GRE Score'].mean())
adm_df['TOEFL Score'] = adm_df['TOEFL Score'].fillna(adm_df['TOEFL Score'].mean())
adm_df['University Rating'] = adm_df['University Rating'].fillna(adm_df['University Rating'].mean())

y = adm_df['Chance of Admit']
X = adm_df.drop(columns='Chance of Admit')

ts = 0.10
rs = 12

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=ts, random_state=rs)

with mlflow.start_run():
    lr = LinearRegression().fit(X_train, y_train)
    adm_pred = lr.predict(X_test)

    rmse, mae, r2 = eval_metrics(y_test, adm_pred)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param('test size', ts)
    mlflow.log_param("random state", rs)
    mlflow.log_metric('RSME', rmse)
    mlflow.log_metric('MAE', mae)
    mlflow.log_metric('R2', r2)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(lr, "model", registered_model_name="LinearregressionAdmissionModel")
    else:
        mlflow.sklearn.log_model(lr, "model")
