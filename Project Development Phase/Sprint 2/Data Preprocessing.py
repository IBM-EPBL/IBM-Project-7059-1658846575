import glob
from PIL import Image
import numpy as np
from keras.applications import *
from keras.preprocessing import *
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split



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
