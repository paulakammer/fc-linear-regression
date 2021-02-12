from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df = pd.read_csv("/home/spaethju/Projects/FeatureCloud/fc_apps/fc-linear-regression/sample_data/cancer_reg/cancer_reg.csv").select_dtypes(include=numerics).dropna()

X = df.drop("TARGET_deathRate", axis=1)
y = df.loc[:, "TARGET_deathRate"]

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = LinearRegression()
reg.fit(X, y)

y_pred = reg.predict(X_test)

# The mean squared error
scores = {
    "r2_score": [r2_score(y_test, y_pred)],
    "explained_variance_score": [explained_variance_score(y_test, y_pred)],
    "max_error": [max_error(y_test, y_pred)],
    "mean_absolute_error": [mean_absolute_error(y_test, y_pred)],
    "mean_squared_error": [mean_squared_error(y_test, y_pred)],
    "mean_squared_log_error": [mean_squared_log_error(y_test, y_pred)],
    "mean_absolute_percentage_error": [mean_absolute_percentage_error(y_test, y_pred)],
    "median_absolute_error": [median_absolute_error(y_test, y_pred)]
}

print(scores)

