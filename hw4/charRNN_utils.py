from os import walk
from os.path import join

import numpy as np
import math

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,SimpleRNN,Dropout
from keras.optimizers import RMSprop,Adagrad

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import random


def parseFile(path_to_dataset,num_prevChar=30, batchIndex = 0):

    #print('Loading Data ...............................................\n')
  
    # Create List of Unique Characters in the Music    
    fHandle = open(path_to_dataset)
    text = fHandle.read()

    chars=sorted(list(set(text)))
    fHandle.close()

    # Create index number for all the characters
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))    

    # Create training Data X and Y
    sentences = [];     next_chars = [];
    for i in range(0,len(text)-num_prevChar-1):
        sentences.append(text[i:i+num_prevChar])
        next_chars.append(text[i+num_prevChar])


    # if batchIndex == 0:
    #     print("Number of characters in data file: ",len(text))
    #     print('Vocab length :\t',len(chars))
    #     print('Total number of batches: \t',len(sentences))

    startIndex  =  int(len(sentences)/8) *  batchIndex
    endIndex    =  int(len(sentences)/8) * (1+batchIndex)

    # print "startIndex: ",startIndex
    # print "endIndex:   ",endIndex
    sentences   =  sentences[startIndex:endIndex]
    next_chars  =  next_chars[startIndex:endIndex]

    
    # print('Vectorization..............')
    X = np.zeros((len(sentences), num_prevChar, len(chars)))
    y = np.zeros((len(sentences), len(chars)))



    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i,char_indices[next_chars[i]]] = 1;
    
    trainData = {}
    testData = {}
    
    [ trainData['x'], testData['x'], trainData['y_true'],testData['y_true']]  =  train_test_split(X, y, test_size=0.2)    
	 
    # print('Number of Training Examples: \t',trainData['x'].shape[0])
    # print('Number of Test Examples: \t', testData['x'].shape[0])
    
    vocab = {}
    vocab['char_2_indices'] = char_indices
    vocab['indices_2_char'] = indices_char
    vocabLength = len(chars)
    seedList = sentences

    #print("Batch %d processed", batchIndex)
    # print('\nComplete.')
    return(trainData,testData, vocab, vocabLength, seedList)



def buildModel(num_prevChar = 25,vocabLength=100,nHiddenNeuron=100,rnn_lstm='rnn',	percentDropout=0,optimizerUsed='RMSprop'):
    print('\nBuilding model.......................................')
    model = Sequential()
    
    if(rnn_lstm == 'rnn'):
    	model.add(SimpleRNN(nHiddenNeuron,input_shape=(num_prevChar, vocabLength), return_sequences=False))
    if(rnn_lstm == 'lstm'):
    	model.add(LSTM(nHiddenNeuron,input_shape=(num_prevChar, vocabLength), return_sequences=False))
    
    #model.add(LSTM(nHiddenNeuron,return_sequences=True))
    #model.add(LSTM(nHiddenNeuron,return_sequences=False))

    model.add(Dropout(percentDropout))
    model.add(Dense(vocabLength,activation='softmax'))
    
    if(optimizerUsed == 'RMSprop'):
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01,decay=0),metrics=['acc'])
    if(optimizerUsed == 'Adagrad'):
        model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.1,decay=0.1),metrics=['acc'])
    
    #print('Dropout Percentage: ',percentDropout,'%')
    #print('Optimizer Used: ',optimizerUsed)
    print('Model Build.')
    model.summary()
    return(model)


def generateSequence(fHandle, model,num_prevChar, vocabLength, vocab, maxLength,seedSentence, count = 1):

    generatedSequence = seedSentence
    fHandle.write(str(count)+'. \n\n')
    fHandle.write('Seed Sentence: '+str(seedSentence)+'\n\n')

    char_indices = vocab['char_2_indices']
    indices_char = vocab['indices_2_char']

    # Generate Sentence
    for i in range(maxLength):
        predict_next_char = predictNextChar(model,num_prevChar,vocabLength,seedSentence,char_indices,indices_char);
        generatedSequence = generatedSequence + predict_next_char
        seedSentence = seedSentence[1:] + predict_next_char
    fHandle.write('Generated Sequence: \n'+str(generatedSequence)+'\n\n\n')
    
    
def predictNextChar(model,num_prevChar,vocabLength,sentence,char_indices,indices_char):
    X = np.zeros((1,num_prevChar,vocabLength))

    for i,c in enumerate(sentence):
        X[0,i,char_indices[c]] = 1

    pred = model.predict(X,verbose = 0)[0]
    preds = np.asarray(pred).astype('float64')
    preds = preds/preds.sum()
    #char_predict = indices_char[np.argmax(preds)]
    probas = np.random.multinomial(1, preds, 1)
    char_predict = indices_char[np.argmax(probas)]
    return(char_predict)




def plotGraph(history, percentDropout = 0, nHiddenNeuron= 0,optimizerUsed = 'RMSprop'):
    plt.plot(history.history['loss'],'r-', label='Train Loss')
    plt.plot(history.history['val_loss'],'b-', label='Validation Loss')
    plt.tick_params(labelright = True)
    plt.title('"Train/Validation Loss vs Epoch"')
    plt.ylabel('Train/Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left', shadow=True)
    
    xCoord = int(0.5*len(history.history['acc']));
    ran = max(history.history['loss']+history.history['val_loss']) - min(history.history['loss']+history.history['val_loss'])
    st = min(history.history['loss']+history.history['val_loss'])
    
    # plt.text(xCoord,st+ran*0.85, 'Dropout : '+str(percentDropout))
    # plt.text(xCoord,st+ran*0.9,'Neurons : '+str(nHiddenNeuron) )
    # plt.text(xCoord,st+ran*0.95, 'Optimier: '+optimizerUsed )
    
    # fileName = 'trainPlot_Dropout_'+str(percentDropout)+'_Neuron_'+str(nHiddenNeuron)+'_'+optimizerUsed +'.jpg'
    # print('Filename = ',fileName)
    plt.show()
    # plt.savefig(fileName)