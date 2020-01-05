
from sklearn.linear_model import LinearRegression
import pandas as pd
import pylab as plt
import numpy.random as nprnd
import random
import matplotlib.pyplot as plt


# load data 
df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# explore 
df.head()

df.boxplot()

from pandas.tools.plotting import scatter_matrix 

scatter_matrix(df, alpha=0.2, figsize=(7,7), diagonal='kde')


# fit linear model 
from sklearn.linear_model import LinearRegression

# define dependent and intependent variables 
Y = df['sales']
X = df[['TV','radio','newspaper']]

X.insert(0, 'intercept', 1.0)    # add one more column for intercept value

X.head()
X.shape


# split the data 80% training 20% testing
from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.8)

X_test.shape


# train the model 
import numpy as np
from numpy.linalg import inv

# approach 1 
XT = X_train.transpose()
Beta= inv(XT.dot(X_train)).dot(XT).dot(Y_train)
Beta

# approach 2 
lm = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
print ( 'lm.intercept_ : ', lm.intercept_)
print ( 'lm.coef_ : ', lm.coef_ )


# find R^2 fir train and test and compare. 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

Y_pred_train = lm.predict(X_train)
Y_pred_test = lm.predict(X_test)

print('train_R^2 score : %.2f' % r2_score(Y_train, Y_pred_train))
print('test_R^2 score : %.2f' % r2_score(Y_test, Y_pred_test))


# plot real against prediction color coded by data pocket (train, test)
plt.scatter(Y_pred_train,Y_train, c="#1c9acc", alpha=0.5 , s = 60 , edgecolors='none', label="train")
plt.scatter(Y_pred_test,Y_test, c="r", alpha=0.5 , s = 60 ,edgecolors='none', label="test")
plt.xlabel('Predicted Sales')
plt.ylabel('Real Sales')
plt.legend(loc= 2 ,fontsize = 'small' )


# compare coefficients
print('TV : %.4f' % lm.coef_[1])
print('Radio : %.4f' % lm.coef_[2])
print('newspaper : %.4f' % lm.coef_[3])

#NOTE: Radio seems to have the highest correlation coefficient followed by TV. There is a negligible correlation with between newspaper and Sales in the presence of other channels of advertising.


# coefficients confidance interval 
import scipy, scipy.stats
import statsmodels.formula.api as sm


result = sm.OLS(Y_train, X_train).fit()
result.summary()
result.conf_int()

# calculate the error bar
errbar= result.params - result.conf_int()[0]

# plotting 
errbar= result.params - result.conf_int()[0]

coef_data = pd.DataFrame({'coef': result.params.values[1:],
                          'err': errbar.values[1:],
                          'varname': errbar.index.values[1:] })

plt.errorbar(range(len(coef_data['varname'])), coef_data['coef'], yerr=coef_data['err'], fmt='o')
plt.xlim(-1,3)
plt.xticks(range(len(coef_data['varname'])), coef_data['varname'])

plt.show()

plt.bar([1,2,3], coef_data['err'])

plt.xticks([1.5,2.5,3.5], ('TV', 'radio', 'newspaper'))
plt.ylim(0,0.025)

coef_data




