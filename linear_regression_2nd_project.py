import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([[1], [2], [3], [4], [5]])  
y = np.array([2.1, 4.0, 6.2, 8.1, 9.8])  

model = LinearRegression()
fit = model.fit(X, y)
predictions = model.predict(X)

coef = model.coef_
intercept = model.intercept_
print('coefficient:',coef)
print('intercept:',intercept)
print('predictions:',predictions)
r2 = r2_score(y, predictions)
print('R squared score:', r2)

plt.scatter(X, y, color = 'blue', label = 'Data points')
plt.plot(X, predictions, color = 'red', label = 'Regression line')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression Example')
plt.legend()
plt.show()