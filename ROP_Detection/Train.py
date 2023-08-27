import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
#import cv2
import shutil
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

####################### Parameters #######################
Class_Amounts = 2
Max_Data_Amount = 223 # For Each Class
Augmenting = 1 # 1: Active, 0: Disable
Rounds = 20 # Iterations Of Iterative Training
Epoch_Per_Round = 5
Training_Batch = 69
####################### Loading Data #######################
# Extracting Files Address
# Train Data
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split

Root_Address = ["/Dataset/Train/"]


X_Address = []
Y = []
Classes_Count = []

for j in range(len(Root_Address)):
  Base_Address = os.getcwd() + Root_Address[j]

  Classes = os.listdir(Base_Address)
  Classes.sort()
  print(Classes)
  print("----------------------- -----------------------")

  Temp_Address = []
  for i in range(len(Classes)):
    Classes_Count += [0]
    Class_Address = Base_Address + "/" + str(Classes[i])
    Temp_Address += [Class_Address + "/" + j + "/" for j in os.listdir(Class_Address)]
    Temp_Address.sort()
    print(Class_Address)
    
    while len(Temp_Address) != 0:
      Temp = os.listdir(Temp_Address[0])
      if np.any([".jpg" in name.lower() for name in Temp]):
        Temp = [Temp_Address[0] + j for j in Temp if ".jpg" in j.lower()]
        Temp.sort()
        Temp = Temp[2:]
        X_Address += Temp
        Y += [Classes[i] for j in range(len(Temp))]
        """
        if "/SELECTED 4010928" in Root_Address[j] or "/selected 4011005/" in Root_Address[j]: # Duplicate Low Quality Data
          X_Address += Temp
          Y += [Classes[i] for j in range(len(Temp))]
        """
        Classes_Count[-1] += len(Temp)
      else:
        Temp_Address += [Temp_Address[0] + j + "/" for j in Temp]
      Temp_Address.pop(0)
      #
  print("%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%")


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Encoding = LabelEncoder().fit(Y)
Y = Encoding.transform(Y)
#Y = Encoding.inverse_transform(Y)
print(Encoding.classes_)

if len(Y.shape) == 1:
  Y = np.expand_dims(Y, axis = 1)

OneHot = OneHotEncoder().fit(Y)
Y = OneHot.transform(Y).toarray()

#X_Address, Y = shuffle(X_Address, Y)
print()


# Test Data
Root_Address = ["/Dataset/Train/"]
#Base_Address = os.getcwd() + "/selected 4011030"
Base_Address = Root_Address[0]

Classes = os.listdir(Base_Address)
Classes.sort()
print(Classes)
print("----------------------- ----------------------- -----------------------")

X_Test_Address = []
Y_Test = []
Classes_Count = []

Temp_Address = []
for i in range(len(Classes)):
  Classes_Count += [0]
  Class_Address = Base_Address + "/" + str(Classes[i])
  Temp_Address += [Class_Address + "/" + j + "/" for j in os.listdir(Class_Address)]
  Temp_Address.sort()
  print(Class_Address)
  
  while len(Temp_Address) != 0:
    Temp = os.listdir(Temp_Address[0])
    if np.any([".jpg" in name.lower() for name in Temp]):
      Temp = [Temp_Address[0] + j for j in Temp if ".jpg" in j.lower()]
      Temp.sort()
      Temp = Temp[2:]
      X_Test_Address += Temp
      Y_Test += [Classes[i] for j in range(len(Temp))]
      Classes_Count[-1] += len(Temp)
    else:
      Temp_Address += [Temp_Address[0] + j + "/" for j in Temp]
    Temp_Address.pop(0)
    #
    
# Loading Test Data From Extracted Files Address

X_Test = []

Resize_Shape = (256, 256)

for i in range(len(X_Test_Address)):
  X_Test += [np.array(Image.open(X_Test_Address[i]).resize(Resize_Shape, Image.ANTIALIAS))]

#X = np.array(X, dtype = np.float32) / 255.0
X_Test = np.array(X_Test, dtype = np.float16) / 255.0
Y_Test = Encoding.transform(Y_Test)

if len(Y_Test.shape) == 1:
  Y_Test = np.expand_dims(Y_Test, 1)
Y_Test = OneHot.transform(Y_Test).toarray()
print()

if "Class_Amounts" not in globals() or "Class_Amounts" not in locals():
  Class_Amounts = len(Classes)

####################### Model #######################
Model_Name = "Model_20230827"

#Input_Shape = (256, 256, 3)
#L_0 = tf.keras.layers.Input(shape = Input_Shape)
L_0 = tf.keras.layers.Input(shape = X_Test.shape[1:])

L_1 = tf.keras.layers.Conv2D(16, (3, 3), activation = "relu", padding = "same")(L_0)
L_1 = tf.keras.layers.BatchNormalization()(L_1)
L_1 = tf.keras.layers.Conv2D(16, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_1)

L_2 = tf.keras.layers.Conv2D(16, (3, 3), activation = "relu", padding = "same")(L_1)
L_2 = tf.keras.layers.BatchNormalization()(L_2)
L_2 = tf.keras.layers.Conv2D(16, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_2)

L_3 = tf.keras.layers.Conv2D(16, (3, 3), activation = "relu", padding = "same")(L_2)
L_3 = tf.keras.layers.BatchNormalization()(L_3)
L_3 = tf.keras.layers.Conv2D(24, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_3)

L_4 = tf.keras.layers.Conv2D(24, (3, 3), activation = "relu", padding = "same")(L_3)
L_4 = tf.keras.layers.BatchNormalization()(L_4)
L_4 = tf.keras.layers.Conv2D(24, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_4)

L_5 = tf.keras.layers.Conv2D(24, (3, 3), activation = "relu", padding = "same")(L_4)
L_5 = tf.keras.layers.BatchNormalization()(L_5)
L_5 = tf.keras.layers.Conv2D(24, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_5)

L_6 = tf.keras.layers.Conv2D(24, (3, 3), activation = "relu", padding = "same")(L_5)
L_6 = tf.keras.layers.BatchNormalization()(L_6)
L_6 = tf.keras.layers.Conv2D(32, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_6)

L_7 = tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", padding = "same")(L_6)
L_7 = tf.keras.layers.BatchNormalization()(L_7)
L_7 = tf.keras.layers.Conv2D(32, (3, 3), strides = (2, 2), activation = "relu", padding = "same")(L_7)

L_Out = L_7

L_Out = tf.keras.layers.Flatten()(L_Out)
L_Out = tf.keras.layers.Dense(Class_Amounts, activation = "softmax")(L_Out)


Model = tf.keras.Model(inputs = L_0, outputs = L_Out)

print("Model Parameters:", Model.count_params(), "\tLayers:", len(Model.layers))

tf.keras.utils.plot_model(Model, show_shapes = True
                          , show_dtype = True, show_layer_names = True
                          , show_layer_activations = True, to_file = os.getcwd() + "/Model_Structure.png")


Model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = "Adam", metrics = ["Acc", "MSLE"]) # , "MSE", "MAE"


####################### Training #######################

#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split

Temp = []
for i in range(len(Classes)):
  Temp += [len([j for j in range(len(X_Address)) if np.argmax(Y[j]) == i])]

#Data_Amount = np.min([int(0.71 * np.min(Temp)), 223])
Data_Amount = np.min([int(0.71 * np.min(Temp)), Max_Data_Amount])

Resize_Shape = (256, 256)
History = pd.DataFrame([])

if os.path.isdir(os.getcwd() + "/Model/" + Model_Name + "/") and os.path.isfile(os.getcwd() + "/Model/" + Model_Name + "/saved_model.pb"):
  print("Model Exist => Loading...")
  Model = tf.keras.models.load_model(os.getcwd() + "/Model/" + Model_Name + "/")
  History = pd.read_csv(os.getcwd() + "/Model/" + Model_Name + "/History.csv")
  History = History.drop(columns = History.keys()[0])
  print("Model Was Loaded!!")


for i in range(Rounds):
  print("Round", i+1, "/", Rounds)
  
  # Balanced Data
  X_Address, Y = shuffle(X_Address, Y) # , random_state = 23
  X_Train = []
  Y_Train = []
  for i in range(len(Classes)):
    Temp = [j for j in range(len(X_Address)) if np.argmax(Y[j]) == i][:Data_Amount]
    X_Train += np.array(X_Address)[Temp].tolist()
    Y_Train += Y[Temp].tolist()
  Y_Train = np.array(Y_Train)
  #
  print("Loading Data...")
  X = []
  for i in range(len(X_Train)):
    X += [np.array(Image.open(X_Train[i]).resize(Resize_Shape, Image.ANTIALIAS))]

  X = np.array(X, dtype = np.float16) / 255.0
  print("Data Is Just Loaded!!")
  #
  # Augmentation
  if Augmenting == 1:
    print("Augmentation")
    X_2 = np.flip(X, axis = 1) # UpDown
    X_3 = np.flip(X, axis = 2) # Left To Right
    X_4 = np.rot90(X, k = 1, axes = (1, 2)) # k = 1: 1 Time That Means 90 Degrees,
    X_5 = np.rot90(X, k = 3, axes = (1, 2)) # k = 3: 3 Time That Means 270 Degrees,
    X_6 = np.flip(X_3, axis = 1) # UpDown
    X_7 = np.flip(X_4, axis = 2) # Left To Right
    X_8 = np.flip(X_5, axis = 1) # UpDown
    X_9 = tf.image.adjust_contrast(X, 2.3).numpy() # Increase Contrast By Factor 2.3

    X_Aug = np.concatenate((X, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9))
    Y_Aug = np.concatenate(tuple([Y_Train for i in range(int(X_Aug.shape[0] / X.shape[0]))])) # np.transpose(np.tile(np.transpose(Y), 8))
    del X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9
    X = X_Aug
    Y_Train = Y_Aug
    del X_Aug, Y_Aug
  #
  X_Train = np.copy(X)
  del X
  #
  Model.fit(X_Train, Y_Train, batch_size = Training_Batch, epochs = Epoch_Per_Round, validation_data = (X_Test, Y_Test))
  #
  # History
  if len(Model.history.history["loss"]) != 0:
    Temp_History = Model.history.history
    if len(History) != 0:
      History = pd.concat([History, pd.DataFrame(Temp_History, columns = Temp_History.keys())], ignore_index = True)
    else:
      History = pd.DataFrame(Temp_History, columns = Temp_History.keys())
  #

  Model.save(os.getcwd() + "/Model/" + Model_Name + "/")
  History.to_csv(os.getcwd() + "/Model/" + Model_Name + "/History.csv")
  #
  Model = tf.keras.models.load_model(os.getcwd() + "/Model/" + Model_Name + "/")
  #
  print("----------------------- ----------------------- ----------------------- -----------------------")


####################### Classification Report #######################

#from sklearn.metrics import classification_report, confusion_matrix
print("Model Evaluation:", Model.eval(X_Test, Y_Test))
print("::::::::::::::::::::::: ::::::::::::::::::::::: ::::::::::::::::::::::: :::::::::::::::::::::::")
print(confusion_matrix(Y_Test, Model.predict(X_Test)))
print("::::::::::::::::::::::: ::::::::::::::::::::::: ::::::::::::::::::::::: :::::::::::::::::::::::")
print(classification_report(np.argmax(Y_Test, axis = 1), np.argmax(Model.predict(X_Test), axis = 1) ))

"""####################### Loading Data #######################"""

print("Training Is Finished, Hope The Results Were Good Enough.")
