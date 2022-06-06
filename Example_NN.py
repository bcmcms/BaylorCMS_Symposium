#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Written for Python 3.6.9
import pandas as pd ## Using Pandas 1.1.5
from tensorflow import keras ## Tensorflow 1.14.0
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt ## matplotlib 3.3.0

cnames = ['truth','pTl1','etal1','phil1','pTl2','etal2','phil2','MET','METphi','MET_rel','axialMET',
          'M_R','M_TR_2', 'R','MT2','S_R','M_Delta_R','dPhi_r_b', 'cos(theta_r1)']
dataframe = pd.read_csv('SUSY.csv',names=cnames)
trainframe = dataframe[dataframe.index%5 == 0].reset_index(drop=True)
testframe = dataframe[dataframe.index%5 == 1].reset_index(drop=True)

trainx = trainframe.drop('truth',axis=1)
trainy = trainframe['truth']
testx = testframe.drop('truth', axis=1)
testy = testframe['truth']
scaler = MinMaxScaler()
trainx = scaler.fit_transform(trainx)
testx  = scaler.transform(testx)

model = keras.Sequential([
	keras.layers.Dense(18, activation=tf.nn.relu,input_shape=(18,)),
	keras.layers.Dense(18, activation=tf.nn.relu),
	keras.layers.Dense(18, activation=tf.nn.relu),
	keras.layers.Dense(1, activation=tf.nn.sigmoid),
	])
model.compile(keras.optimizers.Adam(learning_rate=0.01),
	loss='binary_crossentropy',
	metrics=['accuracy'])

model.fit(trainx, trainy, batch_size=20000, epochs=40)

rocx, rocy, roct = roc_curve(testy, model.predict(testx).ravel())
trocx, trocy, troct = roc_curve(trainy, model.predict(trainx).ravel())
test_loss, test_acc = model.evaluate(testx, testy)
print('Test accuracy:', test_acc,' AOC: ', auc(rocx,rocy))

plt.clf()
plt.plot([0,1],[0,1],'k--')
plt.plot(rocx,rocy,'red')
plt.plot(trocx,trocy,'b:')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['y=x','Validation','Training'])
plt.title('Keras NN  ROC (area = {:.3f})'.format(auc(rocx,rocy)))
plt.savefig('practice_SUSY_ROC')
