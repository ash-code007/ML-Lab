import numpy as np
import pandas as pd
import csv
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Read the attributes
lines = list(csv.reader(open('data/heart.csv', 'r')));
attributes = lines[0]

#attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
# Read Cleveland Heart disease data
heartDisease = pd.read_csv('data/heart.csv', names = attributes)
heartDisease = heartDisease.replace('?', np.nan)

# Display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())
print('\nAttributes and datatypes')
print(heartDisease.dtypes)

# Model Bayesian Network
model = BayesianModel([ ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'),
                        ('sex', 'trestbps'), ('exang', 'trestbps'), ('trestbps', 'heartdisease'),
                        ('fbs', 'heartdisease'), ('heartdisease', 'restecg'), ('heartdisease', 'thalach'),
                        ('heartdisease', 'chol')])
# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Deducing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n1.Probability of HeartDisease given Age=20')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 40})

print(q['heartdisease'])

print('\n2. Probability of HeartDisease given chol (Cholestoral) =100')

q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'sex': 0, 'chol': 100})
print(q['heartdisease'])