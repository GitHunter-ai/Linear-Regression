import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)



data=pd.read_csv(r"data\advertising.csv")
info=data.head(),data.describe()
print(info)

# Linear Regression for one variable

X=data["TV"].to_numpy().reshape(-1,1)
Y=data["Sales"].to_numpy().reshape(-1,1)
print(X.shape,Y.shape)


# Function for plotting data and results


def plot_regression(X, y, y_pred=None, feature_names=None, title="Regression Plot"):
    """
    Plots regression results for single or multiple features.
    Automatically:
        - For 1 feature: 2D scatter + regression line.
        - For 2+ features: Actual vs Predicted.
    
    Parameters:
    -----------
    X : array-like, shape (m, n_features)
        Input features.
    y : array-like, shape (m,)
        Actual target values.
    y_pred : array-like, shape (m,), optional
        Predicted target values.
    feature_names : list of str, optional
        Names of features (for axis labels).
    title : str
        Title of the plot.
    """
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    if y_pred is not None:
        y_pred = np.array(y_pred).reshape(-1, 1)

    n_features = X.shape[1] if X.ndim > 1 else 1

    # Case 1: Single Feature
    if n_features == 1:
        sorted_idx = np.argsort(X[:, 0])
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='blue', label="Actual data")
        if y_pred is not None:
            plt.plot(X[sorted_idx], y_pred[sorted_idx], color='red', linewidth=2, label="Regression line")
        plt.xlabel(feature_names[0] if feature_names else "Feature")
        plt.ylabel("Target")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    # Case 2: Multiple Features -> Actual vs Predicted
    else:
        if y_pred is not None:
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, color='green', alpha=0.6, label="Predictions")
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Ideal Fit")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("For multiple features, provide y_pred to plot Actual vs Predicted.")


plot_regression(X, Y, feature_names=['TV'], title="TV vs Sales (Raw Data)")

class LinearRegression():
    def __init__(self):       
        self.W=None
        self.b=None
        self.mean=None
        self.std=None
        self.W_org=None
        self.b_org=None
        
    def normalize(self,X):
        self.mean=X.mean(axis=0)
        self.std=X.std(axis=0)
        return (X-self.mean)/self.std
    
    
    
    def initialize_parameters(self,X):
       
        self.W=np.random.randn(X.shape[1],1)*0.01
        self.b=np.zeros((1,1))
        
    def predict(self, X):
        """
        Perform forward propagation for linear regression.

        Arguments:
        X -- input data, shape (m, n_features)

        Returns:
        Y_hat -- predictions, shape (m, 1)
        """
       
        
        Y_hat = X @ self.W + self.b
        return Y_hat

    
    def compute_cost(self,Y,Y_hat):
        m=Y.shape[0]
        cost = 1/(2*m)*np.sum((Y_hat-Y)**2)
        return cost
    
    def fit(self,X,Y,n_iteration,learningrate=0.01,print_cost=False):
        """Trains the model using gradient descent."""
        n=X.shape[0]
        X_norm=self.normalize(X)
        self.initialize_parameters(X)
        
        
        for iteration in range(n_iteration):
            Y_hat =  X_norm @ self.W + self.b
            grad_w=(1/n)*X_norm.T@(Y_hat-Y)
            grad_b=(1/n)*np.sum(Y_hat-Y)
            self.W -= learningrate*grad_w
            self.b -= learningrate*grad_b
            cost = self.compute_cost(Y,Y_hat)
            
            if print_cost and iteration % 50 == 0:
                print(f"cost after {iteration} iteration:{cost}")
                
        self.W_norm = np.array(self.W).copy()
        self.b_norm = np.array(self.b)
        self.denormalize_parameters()
        
        return self.W, self.b

        
    def denormalize_parameters(self):
        mean = np.array(self.mean).reshape(-1, 1)
        std = np.array(self.std).reshape(-1, 1)

        self.W = self.W_norm / std
        self.b = self.b_norm - np.sum((mean / std) * self.W_norm)
        
        return self.W, self.b

    

    def score(self, X, Y):
        """
        Compute R² score on provided data.
        """
        Y_hat = self.predict(X)
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

model=LinearRegression()
model.fit(X,Y,2000,0.01,print_cost=True)
print(f"W:{model.W}\nb:{model.b}")

r2 = model.score(X, Y)
print(f"R² score: {r2:.4f}")
Y_pred = model.predict(X)

plot_regression(X,Y,y_pred=Y_pred,feature_names=["TV"],title="TV vs Sales (Regression Line)")
   

X2=data[["TV","Radio","Newspaper"]].to_numpy().reshape(-1,3)
model2=LinearRegression()
model2.fit(X2,Y,2000,0.01,print_cost=True)
print(f"W:{model2.W}\nb:{model2.b}")

r2 = model2.score(X2, Y)
print(f"R² score: {r2:.4f}")
Y_pred2 = model2.predict(X2)

plot_regression(X2, Y, y_pred=Y_pred2, title="Actual vs Predicted Sales (Multiple Features)")
