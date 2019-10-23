from __future__ import division
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import ELU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, to_categorical
from keras.callbacks import History 
from keras.models import model_from_json
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

#for per class accuracy
from sklearn.metrics import classification_report
# #for balanced accuracy
# from sklearn.metrics import balanced_accuracy_score

#for optimizer
from keras import optimizers

#for custom layer
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from keras.callbacks import EarlyStopping, CSVLogger

import warnings

#from keras import metrics
from keras.metrics import top_k_categorical_accuracy

from sklearn.preprocessing import LabelEncoder

import os

import pickle

import time
import datetime

#tosave report
def classification_report_csv(report,name="report"):
    headers=["class","precision","recall","f1_score","support"]
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-1]:
        row = {}
        row_data = line.split('  ')
        #check length of string:
        if len(row_data) > 1:
          cont = 0
          for i in range(len(row_data)):
            if len(row_data[i]) > 0:
              #this is the real index to start looking!
              row[headers[cont]] = row_data[i]    
              cont = cont+1
          report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    #sort dataframe by precision, byt leave avg as the last row:
    condition=(dataframe["class"] == "avg / total")
    excluded=dataframe[condition]
    included=dataframe[~condition]
    sored = included.sort_values(by=["precision"],ascending=False)
    pd.concat([sored,excluded]).to_csv((name+".csv"), index = False)

start = time.time()

#needed to import from keras 1.2, see here: https://github.com/fchollet/keras/blob/master/keras/legacy/layers.py
#and here for explanation: https://groups.google.com/forum/#!topic/keras-users/rQ2fjaNbX5w

#sequantial model using keras only. output will be a model and also  a history file for politting stuff!

#Initialize Random Number Generator
#This is important to ensure that the results we achieve from this model can be achieved again precisely. It ensures that the stochastic process of training a neural network model can be reproduced.
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#size of inputs
dlsize=2048
bits=2048
#number of layers for highways

cutoffs = [500]

for cont, cutoff in enumerate(cutoffs):

  #load dataset
  file=("ProductBits_top"+str(cutoff)+"rules_"+str(bits)+"bits_test.csv")  
  pathtofile="../sets.with.LargeRC"
  pathtomodel="./"
  print "Loading your data..."
  df = pd.read_csv(os.path.join(pathtofile,file))
  print "Done loading your data!"

  #load values from dataset:
  X_test = df.values[:,3:]
  Y_test = df.values[:,2:3].ravel()

  #DETERMINE RULESIZE BY THE NUMBER OF UNIQUE Y's
  rulesize = len(np.unique(Y_test))

  #integer encode with sklearn
  label_encoder = LabelEncoder()
  integer_encoded_Y = label_encoder.fit_transform(Y_test)
  #one hot encode with keras:
  onehot_Y_test = to_categorical(integer_encoded_Y)

  # #reverse encoding...
  target_names = np.unique(label_encoder.inverse_transform(argmax(onehot_Y_test,axis=1)))
  #convert to strings just in case...
  target_names = [str(i) for i in target_names]

  #def top 5, 10 accuracy:
  def top_5_categorical_accuracy(y_true, y_pred):
  	return top_k_categorical_accuracy(y_true, y_pred, k=5) 
  def top_10_categorical_accuracy(y_true, y_pred):
  	return top_k_categorical_accuracy(y_true, y_pred, k=10) 	

  #save model
  #load from json
  print "Loading your model..."
  model_json = open(os.path.join(pathtomodel,("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules.json")), "r")
  loaded_model_json = model_json.read()
  model_json.close()
  model = model_from_json(loaded_model_json)
  #print("Saved model to disk")  

  #load weights
  model.load_weights(os.path.join(pathtomodel,("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules.h5")))
  print "Model loaded from disk"

  #precision, recall, f1, etc...
  #`preds = model.predict(X_test)` and then compute `precision(preds, y_test)
  Y_pred = model.predict_classes(X_test)# 1-D vector of predictions!
  probs = model.predict(X_test)#: array with probabilities! (this is what we want for markov chain)

  report = classification_report(integer_encoded_Y, Y_pred, target_names=target_names)
  classification_report_csv(report,name="metrics_report_test_set")

  #confustion  matrix:
  cm = confusion_matrix(integer_encoded_Y, Y_pred)

  #sklean: predictios in columns, actual values in rows!
  #for each label build row into this:
  #           | reference 
  # predicted |   class    | not_class
  # class     | row[index] (TP) | sum(row) - row[index] (FP)
  # not_class | sum(column) - row[index] (FN) | sum(matrix) - TP - FN - FP
  #details here: https://www.rdocumentation.org/packages/caret/versions/6.0-78/topics/confusionMatrix

  accs_data = []
  for index,row in enumerate(cm):
    accs = []
    #diagonal value
    TP = row[index]
    #get column
    predictions = cm[:,index]
    #get row
    true = row  
    FP = sum(predictions) - TP
    FN = sum(true) - TP
    #sum of diagonal elements (total TRUE POSITIVES) - TP  
    # TN = sum(np.diagonal(cm))-TP
    # true negatives: total sum of matrix, minus row and column for given class!!
    TN = sum(sum(cm)) - TP - FP - FN
    specificity = float("%.4f" % round(TN/(TN+FP),4))
    sensitivity = float("%.4f" % round(TP/(FN+TP),4))
    balanced_accuracy = "%.4f" % round(0.5*(specificity+sensitivity),4)
    precision = "%.4f" % round(TP/(TP+FP),4) 
    recall = "%.4f" % round(TP/(TP+FN),4)
    accs_data.append([target_names[index],index,TP,FP,FN,TN,balanced_accuracy,specificity,sensitivity,precision,recall])

  #make accs_data into datapframe
  df = pd.DataFrame(accs_data,columns=["Label","LabelInteger","TP","FP","FN","TN","BalancedAccuracy","Specificity","Sensitivity","Precision","Recall"]).sort_values(by="BalancedAccuracy",ascending=False)
  df.to_csv(("balanced_accuracies.csv"), index = False)

  #double checking with R:
  np.savetxt("TrueLabelsForR.txt",integer_encoded_Y)
  np.savetxt("PredictedLabelsForR.txt",Y_pred)

  f = open("balanced_accuracy_score.log",'w')
  f.write("Average Balanced Accuracy: %.4f\n" % np.mean([float(i) for i in df["BalancedAccuracy"].values.tolist()]))
  f.close()

  #Get true probabilities...
  pathout="../probabilities.of.true.label"
  outname=("Probabilities_"+str(cutoff)+"_rule_cutoffs_"+str(bits)+"bits.csv")

  true = argmax(onehot_Y_test,axis=1)
  predicted = Y_pred

  #difficult to track back oriignal input, lets fix that in next version of this script... for now we just need the labels!
  out=[]
  for i in range(len(X_test)):
    within=[]
    #probabilities[]
    #first append test_id, true, predicted, prob_true then all probabilities
    #this is thre true label for sample i
    print true[i]
    #this is the predicted label for sample i
    print predicted[i]
    #this is the probability of the true label of sample i
    print probs[i][true[i]]
    #this is the probability of the predicted label...
    print probs[i][predicted[i]]
    for j in [i,true[i],probs[i][true[i]],predicted[i],probs[i][predicted[i]]]:
      within.append(j)
    for prob in probs[i]:
      within.append(prob)
    out.append(within)

    #make new df with out and probs...
    #columns
  columns=[]
  for i in ["test_id","true_label","true_prob","pred_label","pred_prob"]:
    columns.append(i)
  #for i in range(rulesizes[0]):
  for i in range(rulesize): 
    string=("rule"+str(i+1))
    columns.append(string)  

  df = pd.DataFrame(out,columns=columns)
  df.to_csv(os.path.join(pathout,outname),index=False)  

end = time.time()

f = open((str(datetime.datetime.now()).replace(' ','_')+".log"),'w')
f.write(("It took %.2f secs to run highway_model_"+str(bits)+" script") % (end - start))
f.close()

# matrix  = np.array([[1971, 19, 1, 8, 0, 1], [16, 1940, 2, 23, 9, 10], [8, 3, 1891, 87, 0, 11], [2, 25, 159, 1786, 16, 12], [0, 24, 4, 8, 1958, 6], [11, 12, 29, 11, 11, 1926]])
# metrix = []
# for i, row in enumerate(matrix):
#   TP = row[i]
#   FP = sum(matrix[:,i]) - TP
#   FN = sum(row) - TP
#   #sum of diagonal elements (total TRUE POSITIVES) - TP
#   # TN = sum(np.diagonal(matrix)) - TP
#   TN = sum(sum(matrix))
#   print TP, FP, FN, TN
#   metrix.append([TP,FP,FN,TN])
# metrix = np.array(metrix)  
