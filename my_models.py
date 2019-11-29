import numpy as np

class SimpleLinearRegression:
    
    def __init__(self, a, b, learning_rate):
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
        
    def _grad(self, x, y):
        a = self.a
        b = self.b
        dmse_da = np.mean(2 * (a * x + b - y) * x ) 
        dmse_db = np.mean(2 * (a * x + b - y) * 1) 
        return dmse_da, dmse_db
    
    def _apply_gradients(self, x, y):
        a = self.a
        b = self.b
        learning_rate = self.learning_rate
        
        da, db = self._grad(x, y)
        a = a - learning_rate * da
        b = b - learning_rate * db
        return a, b
        
    def fit(self, x, y):
        """Fit the values of a, b"""
        n_steps = 100
        for i in range(n_steps):
            a, b = self._apply_gradients(x, y)
            self.a = a
            self.b = b
        return self
    
    def predict(self, x):
        """Predict targets for x"""
        return self.a * x + self.b
        