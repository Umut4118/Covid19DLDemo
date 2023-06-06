import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as  ipd
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.callbacks import ModelCheckpoint
from datetime import datetime

filename = 'audio/train/trainnot_sick/audioset___lxDIpII74_10_15.wav'
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)
ipd.Audio(filename)
plt.show()
filename = 'audio/train/trainsick/audioset__0WKVY0n8aE_155_160.wav'
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)
ipd.Audio(filename)
plt.show()


metadata=pd.read_excel('MetadataTrain.xlsx')
mfccs=librosa.feature.mfcc(y=data, sr=sample_rate,n_mfcc=40)


def features_extractor(file):
    audio,sample_rate=librosa.load(filename , res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    filename=os.path.join(os.path.abspath(row["Folder Path"]+'/'+str(row["Name"])))
    final_class_labels=row["Folder Path"]
    data=features_extractor(filename)
    extracted_features.append([data,final_class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','path'])
extracted_features_df.head()
X=np.array(extracted_features_df['feature'].tolist())
Y=np.array(extracted_features_df['path'].tolist())
y=np.array(pd.get_dummies(Y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#model tasarımı
num_labels=y.shape[1]
model=Sequential()
### First Layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
### Second Layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
### Third Layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
### Final Layer
model.add(Dense(num_labels))
model.add(Activation(tf.nn.softmax))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

##Training model
num_epochs=100
num_batch_size=32
chechpointer=ModelCheckpoint(filepath='saved_models/audio_class.hdf5',verbose=1,save_best_only=True)
start=datetime.now()
model.fit(X_train,y_train,batch_size=num_batch_size,epochs=num_epochs,validation_data=(X_test,y_test),callbacks=[chechpointer])
duration=datetime.now() - start
print("in Time",duration)
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

