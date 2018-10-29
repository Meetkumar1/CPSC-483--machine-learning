import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from math import sqrt

# Read dataset from file
dataset = pd.read_csv('Dataset.csv')
numpyData = np.array(dataset)

# Separate independent and dependent columns
X = numpyData[:,:4]
Y = numpyData[:,-1]

# Splitting dataset into training  set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


'''

 LINEAR REGRESSION

'''

#Fitting multi-variate Linear regression to the Training set
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)

print "The Linear model in mathematical equation -"
print "idx = Tm({0}) + Pr({1}) + Th({2}) + Sv({3}) + {4}".format(linear_regressor.coef_[0],linear_regressor.coef_[1],linear_regressor.coef_[2],linear_regressor.coef_[3],linear_regressor.intercept_)


#Prediction for Test Set
Y_pred = linear_regressor.predict(X_test)

#Root mean square for Linear regression
rms_linear = sqrt(mean_squared_error(Y_test, Y_pred))

# Arrays to store the RMSE for each method
X_plot = np.array(["Linear","Order 2", "Order 3", "Order 4", "Order 5", "Order 6", "Order 7", "Order 8"])
Y_plot = np.array([])
Y_plot = np.append(Y_plot,rms_linear)



'''

 Non linear regression using Polynomial Order 2 to 8

'''

for deg in range(2,9):
    poly_regressor = PolynomialFeatures(degree = deg)
    X_poly = poly_regressor.fit_transform(X_train)
    
    linear_regressor_poly = LinearRegression()
    linear_regressor_poly.fit(X_poly, Y_train)
    
    # Prediction for test set
    Y_pred = linear_regressor_poly.predict(poly_regressor.fit_transform(X_test))
    
    #Root mean square for Linear regression
    rms_non_linear = sqrt(mean_squared_error(Y_test, Y_pred))
    print rms_non_linear
    
    #Append to Plotting array Y axis
    Y_plot = np.append(Y_plot,rms_non_linear)




'''

 Uncertainty analysis

'''

plt.ylabel('RMS Error')
plt.plot(X_plot,Y_plot)

# Above graph shows that the Uncertainity decreases uptill Order 5, but increases after due to Overfitting


# To get Mathematical model for Non linear equation with degree 2

poly_regressor = PolynomialFeatures(degree = 2)
X_poly = poly_regressor.fit_transform(X_train)
    
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(X_poly, Y_train)

print "The Polynomial Order 2 in mathematical equation -"
print "idx = {0} + ({0})Tm^1 + ({1})Tm^2 + ({2})Pr^1 + ({3})Pr^2 + ({4})Th^1 + ({5})Th^2 + ({6})Sv^1 + ({7})Sv^2".format(linear_regressor_poly.intercept_,linear_regressor_poly.coef_[1],linear_regressor_poly.coef_[2],linear_regressor_poly.coef_[3],linear_regressor_poly.coef_[4],linear_regressor_poly.coef_[5],linear_regressor_poly.coef_[6],linear_regressor_poly.coef_[7],linear_regressor_poly.coef_[8],linear_regressor_poly.coef_[9],linear_regressor_poly.coef_[10],linear_regressor_poly.coef_[11],linear_regressor_poly.coef_[12],linear_regressor_poly.coef_[13],linear_regressor_poly.coef_[14])