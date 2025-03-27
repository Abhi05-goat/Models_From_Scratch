# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:39:44 2025
@author: asiva
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Salary_dataset.csv')

# Select relevant columns (removing any unnecessary ones)
df = df.iloc[:, 1:]

# Separate independent (X) and dependent (y) variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Visualizing the data
plt.scatter(X, y, c='b', s=4, label="Years_Exp")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

# Function to calculate Mean Squared Error (MSE) loss
def loss_function(weight, bias, points):
    total_error = 0
    n = len(points)
    
    for i in range(n):
        X = points.iloc[i]['YearsExperience']
        y = points.iloc[i]['Salary']
        total_error += (y - (weight * X + bias)) ** 2
    
    return (0.5) * (1 / n) * total_error  # Return computed loss

# Function for gradient descent update
def gradient_descent(weight_now, bias_now, points, L):
    weight_gradient = 0
    bias_gradient = 0
    n = len(points)
    
    for i in range(n):
        X = points.iloc[i]['YearsExperience']
        y = points.iloc[i]['Salary']
        
        weight_gradient += (-1 / n) * X * (y - (weight_now * X + bias_now))
        bias_gradient += (-1 / n) * (y - (weight_now * X + bias_now))
    
    # Update weight and bias
    weight = weight_now - L * weight_gradient
    bias = bias_now - L * bias_gradient
    
    return weight, bias

# Initialize model parameters
weight = 0
bias = 0
learning_rate = 0.0001  # Step size for gradient descent
epochs = 1000  # Number of iterations

# Lists to store values for visualization
losses = []
weights = []
biases = [] 

# Perform Gradient Descent
for i in range(epochs):
    loss = loss_function(weight, bias, df)
    weight, bias = gradient_descent(weight, bias, df, learning_rate)
    
    # Store values for visualization
    if weight != 0 and bias != 0:
        losses.append(loss)
        weights.append(weight)
        biases.append(bias)

# Plot the fitted regression line
plt.scatter(df['YearsExperience'], df['Salary'], c='b', s=4)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.plot([weight * X + bias for X in range(1, 12)], c='r', label="Regression Line")
plt.legend()
plt.show()      

# Plot cost function vs. weight
plt.scatter(weights, losses, c='y', s=4)
plt.xlabel('Weight')
plt.ylabel('Cost (MSE)')
plt.grid(True)
plt.show()  

# Plot cost function vs. bias
plt.scatter(biases, losses, c='g', s=4)
plt.xlabel('Bias')
plt.ylabel('Cost (MSE)')
plt.grid(True)
plt.show()
