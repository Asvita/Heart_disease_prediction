
# coding: utf-8

# # Heart Disease Prediction Models using Decision Tree with Classification and Regression in Apache Spark

# ## Initialize Spark Context

# In[1]:

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local[*]").setAppName("heart-disease-prediction-descision-tree")
sc   = SparkContext(conf=conf)

print "Running Spark Version %s" % (sc.version)


# ## Read the UCI Heart Disease [Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) and Clean-up

# In[120]:

import pandas as pd
import numpy as np


heartdf = pd.read_csv("processed.cleveland.data",header=None)

print "Original Dataset (Rows:Colums): "
print heartdf.shape
print 

print "Categories of Diagnosis of heart disease (angiographic disease status) that we are predicting"
print "-- Value 0: < 50% diameter narrowing"
print "-- Value 1: > 50% diameter narrowing "
print heartdf.ix[:,13].unique() #Column containing the Diagnosis of heart disease

newheartdf = pd.concat([heartdf.ix[:,13], heartdf.ix[:,0:12]],axis=1, join_axes=[heartdf.index])
newheartdf.replace('?', np.nan, inplace=True) # Replace ? values

print
print "After dropping rows with anyone empty value (Rows:Columns): "
ndf2 = newheartdf.dropna()
ndf2.to_csv("heart-disease-cleaveland.txt",sep=",",index=False,header=None,na_rep=np.nan)
print ndf2.shape
print ndf2.ix[:5,:]


# ## Create a Labeled point which is a local vector, associated with a label/response

# In[123]:

from pyspark.mllib.regression import LabeledPoint

points = sc.textFile('heart-disease-cleaveland.txt') 

def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    values = [float(s) for s in line.strip().split(',')]
    if values[0] == -1: # Convert -1 labels to 0 for MLlib
        values[0] = 0
    elif values[0] > 0:
        values[0] = 1
    return LabeledPoint(values[0], values[1:])

parsed_data = points.map(parsePoint)

print 'After parsing, number of training lines: %s' % parsed_data.count()

parsed_data.take(5)


# ## Perform Classification using a Decision Tree

# Train a decision tree model with Gini impurity as an impurity measure and a maximum tree depth of 3.

# In[126]:

from pyspark.mllib.tree import DecisionTree
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = parsed_data.randomSplit([0.7, 0.3])
# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=5, categoricalFeaturesInfo={}, impurity='gini', maxDepth=3, maxBins=32)


# ### Evaluate model on test instances and compute test error

# In[127]:

predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())


# ## Perform Regression using a Decision Tree

# Train a decision tree regression model with variance as an impurity measure and a maximum tree depth of 5

# In[129]:

model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={}, impurity='variance', maxDepth=3, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression tree model:')
print(model.toDebugString())


# In[ ]:



