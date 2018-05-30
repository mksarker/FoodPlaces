"""
@author: Md Mostafa Kamal Sarker
@ Department of Computer Engineering and Mathematics, Universitat Rovira i Virgili, 43007 Tarragona, Spain
@ email: m.kamal.sarker@gmail.com
@ Date: 23.05.2017
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
## import keras library
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam, SGD, Adadelta, Adagrad, RMSprop
from keras.callbacks import CSVLogger, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from sklearn.utils import class_weight 
## pre-trained models from keras
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
## import evaluation metrics
### import scikit learn libraries
from sklearn.metrics import classification_report,confusion_matrix
from confusion_matrix import plot_confusion_matrix
from sklearn.metrics import f1_score
## data load from data_loder.py
from data_loader import load_images, ix_to_class, nub_class
## ignore warnings
import warnings
warnings.filterwarnings("ignore")

######################### Load train and test images ##########################################
X_train, y_train = load_images('/media/mostafa/Data/Food/FoodPlaces/data/train')
X_val, y_val = load_images('/media/mostafa/Data/Food/FoodPlaces/data/val')
# print (X_train.shape)
######################### convert data type ####################################################
X_val = X_val.astype('float32')
X_train = X_train.astype('float32')
############################### Mean ###########################################################
X_train -= np.mean(X_train, axis=0)
X_val -= np.mean(X_val, axis=0)
############################# Normalized #######################################################
X_train = X_train.astype('float32') / 255.
X_val = X_val.astype('float32') / 255.
# ######## set class weights for imbalanced classes ##############################################
# use class_weight add inide model.fit (class_weight=class_weight)
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
##################### convert class vector #####################################################
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
############################ MODEL ############################################
input_layer = Input(shape=(299, 299,3), name='image_input')  ## change the input size with model
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)  ## change the models (e.g, VGG, ResNet50)
x = base_model(input_layer)
x = Flatten()(x)
#fine-tune
predictions = Dense(nub_class, activation='softmax', name='predictions')(x)
# this is the model we will train
model = Model(input=input_layer , output=predictions)
########## Model print summary and plot ############################################################
model.summary()
#################   Model compile  #################################################################
## define optimizer and you can change from here
# opt = SGD(lr=0.0001, momentum=0.9)
#opt = Adam(lr=0.0001)
#opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.9)
opt = Adadelta(lr=0.001,rho=0.9, epsilon=1e-06, decay=0.9)
##################
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
################  Model saving callbacks  ###########################################################
### checkpoint for saving model
checkpoint= ModelCheckpoint(monitor='val_top_1_accuracy',filepath='InceptionV3/FoodPlaces.hdf5',
                            verbose=1, save_best_only=True)
### save logger file
csv_logger = CSVLogger('ResNet/EgoFoodPlaces_V1.log')
### EarlyStopping, if the model performance is not update
earlystop=EarlyStopping(monitor='accuracy', min_delta=0, patience=5,verbose=0, mode='max')

# ###learning rate scheduler
def schedule(epoch):
    if epoch < 20:
        return 0.001
    elif epoch < 40:
        return 0.0001
    else:
        return 0.00001
lr_scheduler = LearningRateScheduler(schedule)
################  Model fit  #######################################################################
batch_sz=64
nb_epoch=50
history = model.fit(X_train, y_train, batch_size=batch_sz, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_val, y_val), shuffle=True, class_weight=class_weight,
                    callbacks=[csv_logger, checkpoint])
############### Model evalute  ########################################################################
score = model.evaluate(X_val, y_val, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])
print(y_val[1:2])
#################### Model predict ###################################################################
y_pred = model.predict(X_val)
print(y_pred)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
##############  List all data in history #############################################################
print(history.history.keys())
#################  Summarize history for top_1_accuracy ####################################################
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()
plt.savefig("InceptionV3/train_vs_val_accuracy.png")
############### Summarize history for loss  #########################################################
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig("InceptionV3/train_vs_val_loss.png")
#########################  Evaluate the results ####################################################
# compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
class_names = [ix_to_class[i] for i in range(nub_class)]
print ("\n\n========  Evaluate the results =======================")
############################ Classification Report  #################################################
clf_report=classification_report(np.argmax(y_val,axis=1),
                            y_pred,target_names=class_names)
print ("\nClassification report:\n", clf_report)
########################### Save the results #######################################################
# Save the results
file_perf = open('InceptionV3/performances.txt', 'w')
file_perf.write("CLF_REPORT: " + str(clf_report)
                )
file_perf.close()
############################################ END and ENJPY ###########################################
