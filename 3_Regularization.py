
# rescale (i.e. make features have zero mean and unit variance)
#WHY? The variables are standardized such that we have a fair comparison when we apply restriction to shrink coefficients sizes using LASSO and Ridge methods. The size of these coeeficienct depends on the variable which makes sense to have them rescaled. As a result of this process, we also no longer have an intercept.
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
y_scaled = Y - np.mean(Y)


# split the scaled data 80% training 20% testing
Xs_train,Xs_test,Ys_train,Ys_test = train_test_split(X_scaled,y_scaled,train_size = 0.8)


#----------------- LASSO ----------------- 
from sklearn import linear_model

lm_lasso = linear_model.Lasso(alpha=0.0).fit(Xs_train,Ys_train) # alpha = 0 is basically OLS without penalization
print (lm_lasso.coef_)


# find the best alpha 
alpha = np.linspace(0,1,1000)

# plot alpha agaist R^2
R2 = []
coef = []
for i in alpha:  
    lms = linear_model.Lasso(alpha= i).fit(Xs_train,Ys_train)

    R2.append(lms.score(Xs_test,Ys_test))
    coef.append(lms.coef_ )
    

# scatter plot
axes = fig.add_subplot(1, 1, 1)
plt.plot(alpha,R2 , 'wo')
plt.plot(alpha,R2 , '#7a0870')
#plt.grid()
plt.xlabel('alpha')
plt.ylabel('$R^2$') 
plt.title('LASSO performance given alpha')

# the best alpha is where R^2 is highest 
The_alpha = alpha[np.argmax(R2)]
print 'The best alpha :' , The_alpha


# fit LASSO using the best alpha found in previous step 
ml_ls_test = linear_model.Lasso(alpha= The_alpha).fit(Xs_test,Ys_test)
R2_test_data = ml_ls_test.score(Xs_test,Ys_test)

print 'R^2_test', R2_test_data
print 'coef_test' , ml_ls_test.coef_



#----------------- Ridge ----------------- 
from sklearn.linear_model import Ridge

R2_r = []
coef_r = []

for i in alpha:    
    lmr = linear_model.Ridge(alpha= i).fit(Xs_train,Ys_train)
    R2_r.append(lmr.score(Xs_test,Ys_test))
    coef_r.append(lmr.coef_ )    
    

# scatter plot
axes = fig.add_subplot(1, 1, 1)
plt.plot(alpha,R2_r , 'wo')
plt.plot(alpha,R2_r , '#7a0870')
#plt.grid()
plt.xlabel('alpha')
plt.ylabel('$R^2$')
plt.title('Ridge performance given alpha')


# NOTE:The Ridge method takes the variables in the mode while trying to shrink the model and reduce complexity. However, LASSO methods exclude the unnecessary variables such that coefficient is set to zero, and thus can be used as feature selection model.


