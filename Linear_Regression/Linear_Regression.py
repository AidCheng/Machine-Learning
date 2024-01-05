import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    """
    1. pre-process the raw data
    2. get the number of features
    3. initialise the theta matrix 
    """
    def __init__(self, data, labels,polynomial_degree,sinudsoid_degree, normalize_data = True):
        
        (processed_data, features_mean, features_deviation) = \
            prepare_for_training(data,polynomial_degree,sinudsoid_degree,normalize_data = True)
        
        self.data = processed_data
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinudsoid_degree
        self.normalise_data = normalize_data
        
        features_num = self.data.shape[1] # shape[1]: How many columns in the data
        self.theta = np.zeros((features_num,1)) # initialise a matrix, column 1, line features_num
    
    def train(self,alpha,num_iterations = 500):
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history
        
    def gradient_descent (self,alpha, num_iterations = 500):
        cost_history = []
        for _ in range (num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.loss_function(self.data,self.labels))
        return cost_history
    
    def gradient_step(self,alpha):
        """
        Matrix operations, update parameter theta
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        
        """
        new theta
        """
        theta = theta - alpha * (1/num_examples) * (np.dot(delta.T,self.data)).T
        self.theta = theta
    
    def loss_function(self,data,labels):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]
        
    @staticmethod
    def hypothesis(data,theta):
        prediction = np.dot(data,theta)
        return prediction
    
    def get_cost(self,data,labels):
        processed_data = prepare_for_training \
            (data, self.polynomial_degree, self.sinusoid_degree, self.normalise_data)[0] 
        return self.loss_function(processed_data,labels)
    
    def predict(self,data):
        """
        To predict based on the trained models, testifying our model
        """
        processed_data = prepare_for_training \
            (data, self.polynomial_degree, 
             self.sinusoid_degree, 
             self.normalise_data)[0] 
        predictions = LinearRegression.hypothesis(processed_data, self.theta)
        return predictions