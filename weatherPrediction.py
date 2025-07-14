# %%
import pandas as pd

# load the dataset
df = pd.read_csv("data/auckland_filtered_weather_daily_full.csv")
df.head(), df.columns

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# %%
# drop the 'date' column
df_clean = df.drop(columns=['date'])

print(df_clean.columns.tolist())

# %%
# drop the missing value
df_clean.columns = df_clean.columns.str.strip()

# %%
[col for col in df_clean.columns if 'Temp' in col]

# %%
# clean the dataset
X = df_clean.drop(columns=['Mean Temperature [Deg C]'])
y = df_clean['Mean Temperature [Deg C]']

# %%
# train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)

# %%
#Train Random Forest Regressor

model = RandomForestRegressor(random_state = 42)
model.fit(X_train, y_train)

# %%
#Make predictions
y_pred = model.predict(X_test)
print(len(model.estimators_))

# %%
#plot prediction vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Mean Temperature', color='green', linewidth=2)
plt.scatter(range(len(y_pred)), y_pred, label='Predicted Mean Temperature', color='red', s=10)
plt.title('Actual vs Predicted Mean Temperature (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Mean Temperature [Deg C]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
#Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# %%
#create table to dispaly the metrics
import pandas as pd 

metrics_table = pd.DataFrame({
    "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", 
                "R2 Score"],
    "Value": [mae, mse, rmse, r2] 
    })
display(metrics_table)

# %%

from sklearn.tree import export_text

# Print one tree's logic (text format)
for i, tree in enumerate(model.estimators_):
    print(f"\nTree {i+1}")
    print(export_text(tree, feature_names=list(X_train.columns)))

# %%
# Only print the feature importance
import pandas as pd

importance = model.feature_importances_
features = pd.Series(importance, index=X_train.columns)
features.sort_values(ascending=False).plot(kind='bar', title="Feature Importance")


