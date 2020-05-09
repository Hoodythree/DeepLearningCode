from sklearn.linear_model import LinearRegression
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


lr = LinearRegression()
lr.fit(train[['id']], train[['questions']])
print('Sklearn Coef: %s'%lr.coef_[0][0])
print('Sklearn intercept: %s'%lr.intercept_[0])