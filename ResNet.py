import tensorflow as tf
import keras
from keras import Input, Model
from keras.layers import UpSampling2D, add
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, LeakyReLU, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, rmsprop
from keras.activations import relu, softmax, sigmoid
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable warning message of tensorflow
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.metrics import classification_report,confusion_matrix
import csv
import cv2

# Load X data
path1 = './train_64/'
#X_data = os.listdir(path1)
#X_data = sorted(X_data)
X_data = []
#Y_data = []
Y_data1 = []
Y_data2 = []
Y_data3 = []
#Y_data4 = []

with open('annot_train.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        #if row[2] == '1':
        X_data.append(row[0])
        Y_data1.append(row[1])
        Y_data2.append(row[2])
        Y_data3.append(row[3:13])
        #Y_data1.append(row[2])
        #Y_data2.append(row[3])
        #Y_data3.append(row[4])
        #Y_data4.append(row[5])        
csvFile.close()

X_data = X_data[1:len(X_data)]

X = np.array([np.array(cv2.imread(path1 + str(img),0)).flatten() for img in X_data],'f') 
X = X/255
X = X.reshape(X.shape[0], 64, 64, 1)
X = X.astype('float32')

#Y = Y_data

#Y = np.array(Y[1:len(Y)])
#Y = Y.astype('int64')
Y_data1 = np.array(Y_data1[1:len(Y_data1)])
Y_data2 = np.array(Y_data2[1:len(Y_data2)])
Y_data3 = np.array(Y_data3[1:len(Y_data3)])
Y_data1 = Y_data1.astype('float')
Y_data1 = Y_data1.astype('int')
Y_data1 = Y_data1.astype('int')

X_val = X[0:6000]
X_train = X[6000:len(X)]

Y_data1_val = Y_data1[0:6000]
Y_data1 = Y_data1[6000:len(Y_data1)]

Y_data2_val = Y_data2[0:6000]
Y_data2 = Y_data2[6000:len(Y_data2)]

Y_data3_val = Y_data3[0:6000]
Y_data3 = Y_data3[6000:len(Y_data3)]



##

#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state = 4)
#X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
#X_val = X_val.reshape(X_val.shape[0], 64, 64, 1)
# Assigning X_train and X_test as float
#X_train = X_train.astype('float32') 
#X_val = X_val.astype('float32')

# Normalization of data 
# Data pixels are between 0 and 1
X_train /= 255
X_val /= 255

################################################################################################
def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not
    
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        
        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    
    return f

input_tensor = Input((64, 64, 1))

# first conv2d with post-activation to transform the input data to some reasonable form
x = Conv2D(kernel_size=3, filters=8, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
x = BatchNormalization()(x)
x = Activation(relu)(x)

# F_1
x = block(8)(x)
# F_2
x = block(8)(x)

# F_3
# H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
# and we can't add together tensors of inconsistent sizes, so we use upscale=True
x = block(16, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
# F_4
x = block(16)(x)                     # !!! <------- Uncomment for local evaluation
# F_5
x = block(16)(x)                     # !!! <------- Uncomment for local evaluation

# F_6
x = block(24, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
# F_7
x = block(24)(x)                     # !!! <------- Uncomment for local evaluation

# last activation of the entire network's output
x = BatchNormalization()(x)
x = Activation(relu)(x)

# average pooling across the channels
# 28x28x48 -> 1x48
x = GlobalAveragePooling2D()(x)

# dropout for more robust learning
x = Dropout(0.2)(x)

# last softmax layer
x1 = Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(x)
out1 = Activation(sigmoid)(x1)

x2 = Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(x)
out2 = Activation(sigmoid)(x2)

x3 = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(x)
out3 = Activation(sigmoid)(x3)
#out2 = Activation(sigmoid)(x)
#out3 = Activation(sigmoid)(x)
#out4 = Activation(softmax)(x)

model = Model(inputs=input_tensor, outputs=[out1,out2,out3]) #,out2,out3,out4)
model.compile(loss=['mean_squared_error','binary_crossentropy','binary_crossentropy'], optimizer='Adam', metrics=['accuracy'])

#callbacks = [EarlyStopping(monitor='val_acc', patience=5)]
#results = model.fit(X_train, Y_train, batch_size=512, epochs=20, callbacks=callbacks, validation_data = (X_val, Y_val)) # ,Y_train2,Y_train3,Y_train4,Y_train5,Y_train6,Y_train7,Y_train8,Y_train9], batch_size=512, epochs=20, callbacks=callbacks, validation_split = 0.2)
#model.summary()

def sigmoidal_decay(e, start=0, end=15, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    
    if e > end:
        return lr_end
    
    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))
    
    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end

lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=15))
es = EarlyStopping(monitor='val_loss', patience=5)

hist = model.fit(X_train, [Y_data1,Y_data2,Y_data3] , epochs=15, validation_data =(X_val, [Y_data1_val,Y_data2_val,Y_data3_val]) , batch_size=256, callbacks=[lr, es])
##

# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss'] 
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot( np.argmin(hist.history["val_loss"]), np.min(hist.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();

##
X_test = []
Y_test1 = []
Y_test2 = []
Y_test3 = []

with open('annot_test.csv', 'r') as csvFile:
    reader2 = csv.reader(csvFile)
    for row2 in reader2:
        X_test.append(row2[0])
        #Y_test.append(row2[3:13]) 
        Y_test1.append(row2[1])
        Y_test2.append(row2[2])
        Y_test3.append(row2[3:13])
csvFile.close()

path2 = './test_64/'
X_test = X_test[1:len(X_test)]

Xte = np.array([np.array(cv2.imread(path2 + str(img2),0)).flatten() for img2 in X_test],'f') 
Xte = Xte/255
Xte = Xte.reshape(Xte.shape[0], 64, 64, 1)
Xte = Xte.astype('float')
X_test = Xte

Y_test1 = np.array(Y_test1[1:len(Y_test1)])
Y_test2 = np.array(Y_test2[1:len(Y_test2)])
Y_test3 = np.array(Y_test3[1:len(Y_test3)])
#Y_test = Y_test.astype('float32')
Y_test1 = Y_test1.astype('float')
Y_test2 = Y_test2.astype('int')
Y_test3 = Y_test3.astype('int')


idk = model.evaluate(X_test,[Y_test1,Y_test2,Y_test3])
print(idk)

i=0
temp = X_test[i].reshape(1,64,64,1)
pred = model.predict(temp)
print(Y_test1[i],Y_test2[i],Y_test3[i],'\n')


