#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class MyRidgeRegression:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        a = np.ones([len(X), 1])
        X_tr = np.concatenate([X, a], 1)
        M = np.eye(X.shape[1]+1)
        M[X.shape[1], X.shape[1]] = 0
        self.w = np.linalg.inv(X_tr.T @ X_tr + self.alpha * M) @ X_tr.T @ y
        
    def predict(self, X):
        a = np.ones([len(X), 1])
        X = np.concatenate([X, a], 1)
        y_pred =  X @ self.w
        return y_pred
    
    def get_weights(self):
        return self.w

