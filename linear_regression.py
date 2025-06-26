import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("Housing.csv")
df_info = df.info()

df_head = df.head()

df_shape = df.shape

print(df_info , df_head , df_shape)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())

df_encoded = pd.get_dummies(df , drop_first = True)
print(df_encoded.info())

x = df_encoded.drop('price', axis = 1)
y = df_encoded['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(x_train.shape, x_test.shape ,  y_train.shape , y_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)

intercept = model.intercept_
coeficients = dict(zip(x_train.columns, model.coef_))
print(intercept, coeficients)

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mae, mse, r2)

plt.figure(figsize=(12,10))
plt.scatter(y_test, y_pred , color = 'blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2)

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.grid(True)
plt.tight_layout()
plt.show()





