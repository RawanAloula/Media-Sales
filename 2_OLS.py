
# Seperate model for each feature
# using the same data in multilinear regression (part 1)


fig = plt.figure()
fig.set_figwidth(fig.get_figwidth() * 2)


#----------------- TV ----------------- 
#fit the model
lmTV = LinearRegression(fit_intercept=True).fit(X_train[['TV']], Y_train)
print  'lmTV.intercept : ', lmTV.intercept_
print  'lmTV.coef : ', lmTV.coef_ 

#R^2
Y_pred_train_TV = lmTV.predict(X_train[['TV']])
Y_pred_test_TV = lmTV.predict(X_test[['TV']])
print('TV train_R^2 score : %.2f' % r2_score(Y_train, Y_pred_train_TV ))
print('TV test_R^2 score : %.2f' % r2_score(Y_test, Y_pred_test_TV))

# scatter plot
axes = fig.add_subplot(1, 3, 1)
plt.scatter(Y_pred_train_TV,Y_train, c="#1c9acc", alpha=0.5 , s = 60 , edgecolors='none', label="train")
plt.scatter(Y_pred_test_TV,Y_test, c="r", alpha=0.5 , s = 60 ,edgecolors='none', label="test")
plt.xlabel('Predicted Sales')
plt.ylabel('Real Sales')
plt.title('TV')
plt.legend(loc= 2 ,fontsize = 'small' )

# coef
#print('TV : %.4f' % lmTV.coef_)
print '  '


#----------------- Radio ----------------- 

#fit the model
lmRa = LinearRegression(fit_intercept=True).fit(X_train[['radio']], Y_train)
print  'lmRadio.intercept : ', lmRa.intercept_
print  'lmRadio.coef : ', lmRa.coef_ 

#R^2
Y_pred_train_Ra = lmRa.predict(X_train[['radio']])
Y_pred_test_Ra = lmRa.predict(X_test[['radio']])
print('TV train_R^2 score : %.2f' % r2_score(Y_train, Y_pred_train_Ra))
print('TV test_R^2 score : %.2f' % r2_score(Y_test, Y_pred_test_Ra))

# scatter plot
axes = fig.add_subplot(1, 3, 2)
plt.scatter(Y_pred_train_Ra,Y_train, c="#1c9acc", alpha=0.5 , s = 60 , edgecolors='none', label="train")
plt.scatter(Y_pred_test_Ra,Y_test, c="r", alpha=0.5 , s = 60 ,edgecolors='none', label="test")
plt.xlabel('Predicted Sales')
plt.ylabel('Real Sales')
plt.title('radio')
plt.legend(loc= 2 ,fontsize = 'small' )

# coef
#print('TV : %.4f' % lmRa.coef_)
print '  '



#----------------- Newspaper----------------- 

#fit the model
lmNe = LinearRegression(fit_intercept=True).fit(X_train[['newspaper']], Y_train)
print  'lmNews.intercept_ : ', lmNe.intercept_
print  'lmNews.coef_ : ', lmNe.coef_ 

#R^2
Y_pred_train_Ne = lmNe.predict(X_train[['newspaper']])
Y_pred_test_Ne = lmNe.predict(X_test[['newspaper']])
print('newspaper train_R^2 score : %.2f' % r2_score(Y_train, Y_pred_train_Ne))
print('newspaper test_R^2 score : %.2f' % r2_score(Y_test, Y_pred_test_Ne))

# scatter plot
axes = fig.add_subplot(1, 3, 3)
plt.scatter(Y_pred_train_Ne,Y_train, c="#1c9acc", alpha=0.5 , s = 60 , edgecolors='none', label="train")
plt.scatter(Y_pred_test_Ne,Y_test, c="r", alpha=0.5 , s = 60 ,edgecolors='none', label="test")
plt.xlabel('Predicted Sales')
plt.ylabel('Real Sales')
plt.title('newspaper')
plt.legend(loc= 2 ,fontsize = 'small' )

# coef
#print('newspaper : %.4f' % lmNe.coef_)