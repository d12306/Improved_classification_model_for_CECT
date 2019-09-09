import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D,BatchNormalization,ZeroPadding3D,Dense, Dropout, Flatten, Activation, Input
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))


input_shape = (64,64,64,1)
f = np.load('./003_train.npz')
data_train, labels_train = f['x_train'], f['y_train']
data_test, labels_test = f['x_test'], f['y_test']
print(data_test)
#--------------loading the testing set-----------
#a=open('/shared/xiangruz/classification/rank/rank_testing_003.pickle','rb')
#test_data=pickle.load(a,encoding='latin1')
#data_test = np.asarray(test_data)


#####################################################################
#data_train = np.concatenate((data_train,data_test), axis = 0)
#labels_train = np.concatenate((labels_train, labels_test),axis = 0)
#####################################################################


#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), 64,64,64,1))
data_test = data_test.reshape((len(data_test), 64,64,64,1))
print(labels_train.shape)#42308,13
print(data_train.shape)
print(data_test.shape)
print(labels_test.shape)

perm = np.random.permutation(data_train.shape[0])
data_train = data_train[perm]
labels_train = labels_train[perm]
print(data_train[0])

seed = 42
np.random.seed(seed)

#Create the model
def cnn_model( input_shape, labels=8 ):

    
    model = Sequential()
    '''
    model.add(Conv3D(32, (3, 3,3), activation='relu', input_shape=input_shape, padding='same',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
    #model.add(MaxPooling3D(pool_size=(2, 2,2)))
    model.add(Conv3D(32, (3, 3,3), activation='relu',padding='same',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2,2),strides=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3,3), activation='relu',padding='same',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv3D(64, (3, 3,3), activation='relu',padding='same',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
    
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2,2),strides=(2, 2, 2)))

    model.add(Conv3D(128, (3, 3,3), activation='relu',padding='same',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv3D(128, (3, 3,3), activation='relu',padding='same',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2,2), strides=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='embedding1',init='TruncatedNormal',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.7))
    model.add(Dense(256, activation='relu', name='embedding2',init='TruncatedNormal',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.7))
    model.add(Dense(labels, activation='softmax', name='softmax',init='TruncatedNormal',kernel_regularizer=regularizers.l2(0.001)))
	'''
    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1a',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            input_shape=input_shape))
    #model.add(Conv3D(64, 3, 3, 3, activation='relu',
    #                        padding='same', name='conv1b'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            padding='same', name='conv2a'))
    #model.add(Conv3D(128, 3, 3, 3, activation='relu',
    #                        padding='same', name='conv2b'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
               
                            padding='same', name='conv3a'))
    model.add(Conv3D(256,(3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            padding='same', name='conv3b'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512,( 3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            padding='same', name='conv4b'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.0005),
                            padding='same', name='conv5b'))
    #model.add(BatchNormalization())
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    
    model.add(Flatten())
    #model.add(BatchNormalization())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6',kernel_initializer='TruncatedNormal',kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7',kernel_initializer='TruncatedNormal',kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dropout(.5))
    model.add(Dense(labels, activation='softmax', name='fc8',kernel_initializer='TruncatedNormal'))

    return model

model = cnn_model(input_shape)
#Model parameters
batch_size = 128

# checkpoint
'''
filepath='weights_light.best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]
'''

# 0.003 takes longer and gives about the same accuracy
import keras.optimizers as KOP
kop = KOP.SGD(lr=0.005, decay=1e-7, momentum=0.9, nesterov=True)
#optimizer = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model.load_weights('model_last.h5')
model.compile(optimizer=kop, loss='categorical_crossentropy',  metrics=['accuracy'])

#print_data = model.get_layer('softmax').output
#print(print_data)

model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=200, batch_size=batch_size, shuffle=True)#callbacks=callbacks_list)

#Save the model
model.save('model_003.h5')

#Evaluate
pdb = {0:'4V4A', 1: '1KP8', 2:'2GLS', 3:'5T2C', 4:'2IDB', 5:'3DY4', 6:'1FNT', 7:'1F1B'}
scores = model.predict(data_test)
result_dict= {}
result_list = []
for sub in scores:
    result_dict = {}
    for i in range(len(sub)):
        result_dict[ pdb[i] ] = sub[i]
    result_list.append(result_dict)

f = zip(d.values(),d.keys())
sorted(f)
print(f)

np.save('predict_result_003.npy', np.asarray(f))


#Print accuracy
#print ("Accuracy: %.2f%%" %(scores[1]*100))

