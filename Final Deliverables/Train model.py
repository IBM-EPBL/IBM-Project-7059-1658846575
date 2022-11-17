import os, glob, random
from PIL import Image
import numpy as np
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import *
from keras.preprocessing import *
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Dropout
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split



ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(ROOT_DIR)


def generateListofFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for fol_name in listOfFile:
        fullPath = os.path.join(dirName, fol_name)
        allFiles.append(fullPath)
    return allFiles

def Configure_CNN_Model(output_size):
    K.clear_session()
    model = Sequential()
    model.add(Dropout(0.4,input_shape=(224, 224, 3)))
    model.add(Conv2D(256, (5, 5),input_shape=(224, 224, 3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_size, activation='softmax'))
    return model

def PrepreocessData(subfolders):
    X_data,Y_data,found = [],[],[]
    id_no=0
    for paths in subfolders:
        files = glob.glob (paths + "/*.jpg")
        found.append((paths.split('\\')[-2],paths.split('\\')[-1]))
        

        for myFile in files:
            img = Image.open(myFile)
            img = img.resize((224,224), Image.ANTIALIAS)
            img = np.array(img)
            if img.shape == ( 224, 224, 3):
                X_data.append (img)
                Y_data.append (id_no)
        id_no+=1


    X = np.array(X_data)
    Y = np.array(Y_data)

   
    print("x-shape",X.shape,"y shape", Y.shape)

    X = X.astype('float32')/255.0
    y_cat = to_categorical(Y_data, len(subfolders))

    print("X shape",X,"y_cat shape", y_cat)
    print("X shape",X.shape,"y_cat shape", y_cat.shape)

    return X_data,Y_data,X,y_cat,found; 
def splitData():
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)
    print("The model has " + str(len(X_train)) + " inputs")
    return X_train, X_test, y_train, y_test



augumented_path = ROOT_DIR+"\\Final Deliverables\\Augumented Dataset\\" 
Folders = generateListofFiles(augumented_path)
subfolders = []
for num in range(len(Folders)):
    sub_fols = generateListofFiles(Folders[num])
    subfolders+=sub_fols

X_data,Y_data,X,y_cat,found= PrepreocessData(subfolders)
X_train, X_test, y_train, y_test = splitData()



early_stop_loss = EarlyStopping(monitor='loss', patience=3, verbose=1)
early_stop_val_acc = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
model_callbacks=[early_stop_loss, early_stop_val_acc]

model = Configure_CNN_Model(6)
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
weights = model.get_weights()
model.set_weights(weights)


image_number = random.randint(0,len(X_test))
predictions = model.predict([X_test[image_number].reshape(1, 224,224,3)])

for idx, result, x in zip(range(0,6), found, predictions[0]):
   print("Label: {}, Type : {}, Species : {} , Score : {}%".format(idx, result[0],result[1], round(x*100,3)))



ClassIndex=np.argmax(model.predict([X_test[image_number].reshape(1, 224,224,3)]),axis=1)
print(found[ClassIndex[0]])


model_json = model.to_json() 
with open(ROOT_DIR+"\\Final Deliverables\\DigitalNaturalist.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(ROOT_DIR+"\\Final Deliverables\\DigitalNaturalist.h5")
