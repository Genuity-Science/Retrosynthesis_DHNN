import pandas as pd
import numpy as np
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import ELU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, to_categorical
from keras.callbacks import History 
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

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


from sklearn.utils import class_weight

from sklearn.preprocessing import LabelEncoder

import os

import pickle

import time
import datetime

start = time.time()

#needed to import from keras 1.2, see here: https://github.com/fchollet/keras/blob/master/keras/legacy/layers.py
#and here for explanation: https://groups.google.com/forum/#!topic/keras-users/rQ2fjaNbX5w

class Highway(Layer):
    """Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    """

    def __init__(self,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        #warnings.warn('The `Highway` layer is deprecated '
        #              'and will be removed after 06/2017.')
        if 'transform_bias' in kwargs:
            kwargs.pop('transform_bias')
            warnings.warn('`transform_bias` argument is deprecated and '
                          'has been removed.')
        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='W_carry')
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim,),
                                           initializer='one',
                                           name='b_carry')
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = activations.sigmoid(y)
        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {'init': initializers.serialize(self.init),
                  'activation': activations.serialize(self.activation),
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
layer_count = 5

cutoffs = [500]
#(wc -l USPTOgrantssetunique_with_50_rule_cutoffs.csv) - 1
#rulesizes = [75]

for cont, cutoff in enumerate(cutoffs):

  #rulesize = rulesizes[cont]

  #load dataset
  file=("ProductBits_top"+str(cutoff)+"rules_"+str(bits)+"bits_train.csv")
  #pathtofile="/Users/jbaylon/Scr/retrosynhtesis.test/current.tests.here/DL.with.big.dataset/fp.for.DL"
  pathtofile="../sets.with.LargeRC"
  df = pd.read_csv(os.path.join(pathtofile,file))
 # dataset = df.values

  seed=7
  #of the .8 of data in training set, use .1 of total for validation (.1/.8 = 0.125)
  train, val = train_test_split(df, test_size=0.125, stratify=df["ReactiveCenter"],random_state=seed)

  del df

  X_train = train.values[:,5:]
  #convert column to 1-D vector:
  Y_train = train.values[:,4:5].ravel()

  #SPLIT VAL, TRAIN:
  X_val = val.values[:,5:]
  #convert column to 1-D vector:
  Y_val = val.values[:,4:5].ravel()

  #DETERMINE RULESIZE BY THE NUMBER OF UNIQUE Y's
  rulesize = len(np.unique(Y_train))

  #integer encode with sklearn
  label_encoder = LabelEncoder()
  integer_encoded_Y = label_encoder.fit_transform(Y_train)
  #one hot encode with keras:
  #THIS IS WHAT WE NEED FOR TRAINING THE MODEL
  onehot_Y_train = to_categorical(integer_encoded_Y)

  onehot_Y_val = to_categorical(label_encoder.fit_transform(Y_val))

  #calculate class_weights:
  class_weights = class_weight.compute_class_weight("balanced",np.unique(onehot_Y_train.argmax(1)),onehot_Y_train.argmax(1))

  def array_to_dict(array_class_weights):
    class_weights = {}
    for idx, value in enumerate(array_class_weights):
      class_weights[idx] = value
    return class_weights

  #dict with class_weights
  class_weights = array_to_dict(class_weights)

  #def top 5, 10 accuracy:
  def top_5_categorical_accuracy(y_true, y_pred):
  	return top_k_categorical_accuracy(y_true, y_pred, k=5) 
  def top_10_categorical_accuracy(y_true, y_pred):
  	return top_k_categorical_accuracy(y_true, y_pred, k=10) 	

  #MODEL HERE:
  model = Sequential()
  # ELU is an advance activation:
  elu = ELU(alpha=1.0)
  #First layer, fully-connected (dense layer in keras)
  model.add(Dense(dlsize, input_dim=bits, kernel_initializer="glorot_normal"))
  #activation
  model.add(elu)
  #dropout layer
  model.add(Dropout(0.2))
  #highway layer(s) here...
  for index in range(layer_count):
  	model.add(Highway(activation="relu"))
  #there are 95 different rules that appear more than 50 times
  #model.add(Dense(95))
  model.add(Dense(rulesize))  
  model.add(Activation('softmax'))
  # Compile model
  #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_5_categorical_accuracy, top_10_categorical_accuracy, 'precision', 'recall'])
  #default ADAM learning rate is 0.001, lets try .0001
  adam=optimizers.Adam(lr=0.001)
  #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', top_5_categorical_accuracy, top_10_categorical_accuracy])
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy', top_5_categorical_accuracy, top_10_categorical_accuracy])


  #to save training loss history...
  history = History()
  #call back for early stop when val_loss stops improving...
  #Default based on DBNN model
  earlystop = EarlyStopping(monitor='val_loss',patience=2)
  #log file:
  csv_logger = CSVLogger(("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules.log"))
  #of the .8 of data in training set, use .1 of total for validation (.1/.8 = 0.125)
  #update batch_size form 5 to 10 based on DANN model...
  #need to add class wights
  #hist = model.fit(X_train, onehot_Y_train, epochs=200, batch_size=10, verbose=1, validation_split = 0.125, callbacks=[history,earlystop,csv_logger],class_weight=class_weights)
  #VALIDATE ON STRATIFIED DATA:
  hist = model.fit(X_train, onehot_Y_train, epochs=200, batch_size=10, verbose=1, validation_data = (X_val,onehot_Y_val), callbacks=[history,earlystop,csv_logger],class_weight=class_weights)

  #save model
  #save to json
  model_json = model.to_json()
  with open(("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules.json"), "w") as json_file:
      json_file.write(model_json)
  model.save_weights(("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules.h5"))
  print("Saved model to disk")  

  #save hist using pandas dataframe
  df = pd.DataFrame.from_dict(hist.history)
  df.to_csv(("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules.csv"),index=True)

  #TEST STARTS HERE:""

  # #precision, recall, f1, etc...
  # #`preds = model.predict(X_test)` and then compute `precision(preds, y_test)
  # predictions = model.predict_classes(X_test)# 1-D vector of predictions!
  # probs = model.predict(X_test)#: array with probabilities! (this is what we want for markov chain)
  # #return rules:
  # #label_encoder.inverse_transform(predictions)
  # # 'weighted':
  # # Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). 
  # # his alters "macro" to account for label imbalance; it can result in an F-score that is not between precision and recall.
  # metrics = precision_recall_fscore_support(Y_test.argmax(1),predictions,average="weighted")
  # print "Precision: %4.2f" % metrics[0] 
  # print "Recall: %4.2f" % metrics[1]
  # print "F-beta: %4.2f" % metrics[2]

  # f = open(("highwayModel_"+str(dlsize)+"input_"+str(bits)+"bits_"+str(cutoff)+"_"+str(rulesize)+"rules_metrics.txt"),"w")
  # f.write("Precision: %4.2f \n" % metrics[0])
  # f.write("Recall: %4.2f \n" % metrics[1])
  # f.write("F-beta: %4.2f \n" % metrics[2])
  # f.close()

  #del df, model
  #del X, Y, X_train Y_train, Y_test

#how long does it take to run this script?
end = time.time()

f = open((str(datetime.datetime.now()).replace(' ','_')+".log"),'w')
#f.write("It took %.2f secs to run highway_model_"+str(bits)+" script" % (end - start))
f.write(("It took %.2f secs to run highway_model_"+str(bits)+" script") % (end - start))
f.close()
