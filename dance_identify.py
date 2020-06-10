# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if os.path.isdir(os.path.join(dirname, filename)):
            print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
from keras.preprocessing.image import load_img,img_to_array
from keras.utils import to_categorical
from random import sample
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,recall_score,precision_score,classification_report
from sklearn.decomposition import PCA
from numpy import argmax 

# %%
training_shape = (150,150)


# %%
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if not f.endswith(".jpg"):
                print('{}{}'.format(subindent, f))
            
list_files(r'/kaggle/input')

# %%
train_df = pd.read_csv("/kaggle/input/identify-the-dance-form/train.csv")
test_df = pd.read_csv("/kaggle/input/identify-the-dance-form/test.csv")
train_df.head()

# %%
ser = train_df['target'].value_counts()
plt.figure()
plt.bar(ser.index,ser.values,0.8)
plt.xticks(ser.index,rotation='vertical',color='white')
plt.tight_layout()

# %%
def create_data(df,start_path,target_size):
    df2 = df.copy()
    for i in range(len(df)):
        path = os.path.join(start_path,df.iloc[i,0])
        img = load_img(path,target_size=target_size)
        img = img_to_array(img)
        df2.iat[i,0] = img
    return df2

# %%
def make_dataset(data):
    lis = []
    for array in data['Image']:
        lis.append(array)
    return np.array(lis)

# %%
def make_labels(train_data):
    all_labels = list(train_data['target'].unique())
    y = []
    for value in train_data['target']:
        y.append(all_labels.index(value))
    return to_categorical(y)

# %%
def create_train_and_test_set(train_df,test_df,target_size=training_shape):
    
    train_data = create_data(train_df,'/kaggle/input/identify-the-dance-form/train',target_size)
    test_data = create_data(test_df,'/kaggle/input/identify-the-dance-form/test',target_size)
    
    LABELS = train_data['target'].unique()
    
    X_train = make_dataset(train_data)/255
    X_test = make_dataset(test_data)/255
    y_train = make_labels(train_data)
    
    return X_train,y_train,X_test,LABELS

# %%
X_train,y_train,X_test,LABELS = create_train_and_test_set(train_df,test_df)

# %%
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',y_train.shape)
print('X_test shape : ',X_test.shape)

# %%
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Flatten,Dropout,Conv2D,BatchNormalization
from keras.models import Model,load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.metrics import accuracy
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint
import tensorflow as tf

# %%
pretrained_resnet_model = ResNet50(include_top = False,input_shape=training_shape+(3,),weights = 'imagenet')
for layer in pretrained_resnet_model.layers:
    layer.trainable = False

# %%
pretrained_inceptio_model = InceptionV3(include_top = False,input_shape=training_shape+(3,),weights = 'imagenet')
for layer in pretrained_inceptio_model.layers:
    layer.trainable = False

# %%
#pretrained_inceptio_model.summary()

# %%
def show_example(img,lael,LABELS):
    lael = np.argmax(lael)
    print("Actual label : ",LABELS[lael])
    plt.imshow(img)

# %%
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y_train,test_size=0.2)
print("X_train shape : ",X_tr.shape)
print("Y_train shape : ",y_tr.shape)
print("X_val shape : ",X_val.shape)
print("Y_val shape : ",y_val.shape)

# %%
def create_model_fine_tune(model,lr=1e-4):
    for layer in model.layers:
        layer.trainable = False
        
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True

    conv1 = Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu')(model.output)
    flatten = Flatten()(conv1)
    dense1 = Dense(256,activation = 'relu')(flatten)
    dropout1 = Dropout(0.3)(dense1)
    #dense2 = Dense(32,activation = 'relu')(dropout1)
    #dropout2 = Dropout(0.1)(dense2)
    output = Dense(8,activation='softmax')(dropout1)
    
    model = Model(inputs=model.input,outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr),metrics =['accuracy'])
    
    return model

# %%
model = create_model_fine_tune(pretrained_inceptio_model,lr=1e-8)
#model.summary()

# %%
#learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 10))
history = model.fit(X_tr,y_tr, epochs=100, callbacks=[lr_schedule],verbose=0)
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])

# %%
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1, 0, 30])

# %%
"""
 Optimal lr is 1e-3
"""

# %%
#model = create_model_fine_tune(pretrained_inceptio_model,lr=1e-3)

# %%
def create_model_fine_tune_partially(pretrained_model,lr=1e-3,dense=[1024],conv_filters=[256,256]):
    for layer in pretrained_model.layers:
        layer.trainable = False
        
    for layer in pretrained_model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
    
    x = Conv2D(filters=256,kernel_size=(2,2),strides=1,activation='relu')(pretrained_model.get_layer('mixed5').output)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256,kernel_size=(2,2),strides=2,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(dense[0],activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(8,activation='softmax')(x)
    
    model = Model(inputs=pretrained_model.inputs,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr),metrics =['accuracy'])
    
    return model

# %%
def create_model_retrain(pretrained_model,lr=1e-3,dense=[1024]):
    x = Conv2D(filters=256,kernel_size=(2,2),strides=1,activation='relu')(pretrained_model.get_layer('mixed5').output)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256,kernel_size=(2,2),strides=2,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(dense[0],activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(8,activation='softmax')(x)
    
    model = Model(inputs=pretrained_model.inputs,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr),metrics =['accuracy'])
    
    return model

# %%
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='weighted',zero_division=0)
        _val_recall = recall_score(val_targ, val_predict,average='weighted',zero_division=0)
        _val_precision = precision_score(val_targ, val_predict,average='weighted',zero_division=0)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: {:.2f} — val_precision: {:.2f} — val_recall {:.2f}" .format(_val_f1, _val_precision, _val_recall))

        return

metrics = Metrics()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

# %%
def plot_acc(History):
    fig, ax = plt.subplots(2,1)
    
    ax[0].plot(History.history['loss'], color='b', label="Training loss")
    ax[0].plot(History.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    
    legend = ax[0].legend(loc='best', shadow=True)
    ax[0].set_title('Loss')

    ax[1].plot(History.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(History.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    ax[1].set_title('Accuracy')
    plt.subplots_adjust(bottom=0.25)

# %%
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# %%
train_datagen = ImageDataGenerator(rotation_range=20
                                  ,width_shift_range=0.2
                                  ,height_shift_range=0.2
                                  ,shear_range=0.2
                                  ,zoom_range=0.2
                                  ,horizontal_flip=True)
train_datagen.fit(X_tr)

# %%
EPOCH = 40
BATCH_SIZE = 32
lr = 1e-3

# %%
#fit
#model.fit(X_tr,y_tr,)

# %%
model = create_model_retrain(pretrained_inceptio_model,lr=lr)

# %%
#fit using generator
model.fit(X_tr,y_tr,epochs=1)
history=model.fit_generator(train_datagen.flow(X_tr,y_tr,batch_size=BATCH_SIZE),validation_data=(X_val,y_val),epochs=EPOCH,steps_per_epoch=X_tr.shape[0]//BATCH_SIZE,verbose=2,callbacks=[metrics,es,mc])

# %%
model = load_model('best_model.h5')

# %%
#train accuracy
np.mean(np.argmax(model.predict(X_tr),axis=1) == np.argmax(y_tr,axis=1))

# %%
#validation accuracy
np.mean(np.argmax(model.predict(X_val),axis=1) == np.argmax(y_val,axis=1))

# %%
plot_acc(history)

# %%
y_pred = np.argmax(model.predict(X_val),axis=1)
y_orig = np.argmax(y_val,axis=1)
print(classification_report(y_orig,y_pred,target_names=LABELS))
plot_confusion_matrix(y_orig,y_pred,LABELS,title='Confusion Matrix')

# %%
def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()

# %%
def misclassified_labels(data,pred_labels,true_labels,class1,class2,LABELS):
    LABELS = list(LABELS)
    class1_index = LABELS.index(class1)
    class2_index = LABELS.index(class2)
    print("Original Label : ",class1)
    print('Predicted Label : ',class2)
    
    if true_labels.shape[1] == 8:
        true_labels = np.argmax(true_labels,axis=1)
    if pred_labels.shape[1] == 8:
        pred_labels = np.argmax(pred_labels,axis=1)
    
    actual_class1_ids = np.where(true_labels==class1_index)
    pred_class2_ids = np.where(pred_labels==class2_index)
    
    intersecting_ids = list(actual_class1_ids[0][np.in1d(actual_class1_ids,pred_class2_ids,assume_unique=True)])
    if len(intersecting_ids) > 5 :
        intersecting_ids = sample(intersecting_ids,5)

    #fig,axes = plt.subplots(len(intersecting_ids))
    #for axis,ids in zip(axes,intersecting_ids):
    #    axis.imshow(data[ids])
    figures = {}
    for i in range(len(intersecting_ids)):
        figures[str(i)] = data[intersecting_ids[i]]
    plot_figures(figures,1,len(intersecting_ids))

# %%
misclassified_labels(X_val,model.predict(X_val),y_val,'kathak','bharatanatyam',LABELS)

# %%
def extract_features(model,X):
    if len(X.shape)==4:
        return model.predict(X)
    elif len(X.shape)==3:
        return model.predict(X.reshape((1,X.shape[0],X.shape[1],X.shape[2])))

# %%
"""
## Using extracted features and svm to predict
"""

# %%
train_features = extract_features(pretrained_resnet_model,X_tr)
train_features = train_features.reshape((train_features.shape[0],train_features.shape[1]*train_features.shape[2]*train_features.shape[3]))
val_features = extract_features(pretrained_resnet_model,X_val)
val_features = val_features.reshape((val_features.shape[0],val_features.shape[1]*val_features.shape[2]*val_features.shape[3]))
test_features = extract_features(pretrained_resnet_model,X_test)
test_features = test_features.reshape((test_features.shape[0],test_features.shape[1]*test_features.shape[2]*test_features.shape[3]))

# %%
X_train_svm = train_features
X_val_svm = val_features
y_train_svm = np.argmax(y_tr,axis=1)
y_val_svm = np.argmax(y_val,axis=1)
X_test_svm = test_features

# %%
#reducing feature space
pca = PCA().fit(X_train_svm)
X_train_svm_pca = pca.transform(X_train_svm)
X_val_svm_pca = pca.transform(X_val_svm)

print("Reduced X_tr shape : ",X_train_svm_pca.shape)
print("Reduced X_val shape : ",X_val_svm_pca.shape)

# %%
clf = SVC(kernel='poly',C=0.5,decision_function_shape='ovo').fit(X_train_svm,y_train_svm)
print("Train Score : ",clf.score(X_train_svm,y_train_svm))
print("Val Score : ",clf.score(X_val_svm,y_val_svm))

# %%
from sklearn.svm import SVC
def svc_train(X_tr,y_tr,C_=1):
    clf = SVC(kernel='rbf',C=C_).fit(X_tr,y_tr)
    print(" Score : ",clf.score(X_tr,y_tr))
    return clf

# %%
plot_confusion_matrix(y_val_svm,clf.predict(X_val_svm),LABELS)

# %%
test = clf.predict(X_val_svm)
print(test)

# %%
misclassified_labels(X_val,test,np.argmax(y_val,axis=1),"sattriya","odissi",LABELS)

# %%
def svm_model_select(kernels=['linear', 'poly', 'rbf', 'sigmoid'],C_values=[0.4,0.7,1,3]):
    scores = []
    classifiers = []
    i = 0
    for kernel in kernels:
        print('\n Kernel : ',kernel)
        for C_val in C_values:
            i += 1
            clf2 = SVC(kernel=kernel,C=C_val,decision_function_shape='ovo').fit(X_train_svm,y_train_svm)
            print("Train Score ({}) C = {} : {:.2f}".format(i,C_val,clf2.score(X_train_svm,y_train_svm))) 
            score = clf2.score(X_val_svm,y_val_svm)
            print("Validation score : {:.2f}".format(clf2.score(X_val_svm,y_val_svm)))
            classifiers.append(clf2)
    return classifiers

# %%
classifiers = svm_model_select()

# %%
linear_svms = svm_model_select(kernels=['linear'],C_values=[0.001,0.01,0.02,0.03,0.07,0.1])

# %%
def create_final_soln(model,X_test,test_df,LABELS,out_file='sol.csv'):
    temp = test_df.copy()
    pred = model.predict(X_test)
    if pred.shape[1] == 8 :
        pred = argmax(pred,axis=1)
    pred = [LABELS[i] for i in pred]
    temp['target'] = pred
    temp = temp.set_index(['Image'])
    temp.head()
    temp.to_csv(out_file)

# %%
create_final_soln(model,X_test,test_df,LABELS,'solnn1.csv')

# %%
create_final_soln(classifiers[11],X_test_svm,test_df,LABELS)

# %%
