import tensorflow as tf
import cv2
import numpy as np
import pydot 
import graphviz
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle



def create_model():
    # Input layers
    input_640x480 = tf.keras.Input(shape=(200, 200, 3), name='input_640x480')
    input_50x50 = tf.keras.Input(shape=(20, 20, 3), name='input_50x50')
    input_values = tf.keras.Input(shape=(4,), name='input_values')

    # 640x480 image
    conv1_640 = tf.keras.layers.Conv2D(32, 3, activation='relu')(input_640x480)
    conv2_640 = tf.keras.layers.Conv2D(64, 3, activation='relu')(conv1_640)
    pool_640 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2_640)
    flattened_640 = tf.keras.layers.Flatten()(pool_640)
    dense_640 = tf.keras.layers.Dense(128, activation='relu')(flattened_640)

    #  50x50 image
    conv1_50 = tf.keras.layers.Conv2D(16, 3, activation='relu')(input_50x50)
    conv2_50 = tf.keras.layers.Conv2D(32, 3, activation='relu')(conv1_50)
    pool_50 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2_50)
    flattened_50 = tf.keras.layers.Flatten()(pool_50)
    dense_50 = tf.keras.layers.Dense(64, activation='relu')(flattened_50)

    # Merge  images and values
    merged = tf.keras.layers.concatenate([flattened_640, flattened_50, input_values], axis=-1)
    dense_50 = tf.keras.layers.Dense(64, activation='relu')(merged)
    print("merged",merged)


    # Output layer
    output_values = tf.keras.layers.Dense(4, name='output_values')(dense_50)

    # Create the model
    model = tf.keras.Model(inputs=[input_640x480, input_50x50, input_values], outputs=output_values)
    return model

model=create_model()
print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')
tf.keras.utils.plot_model(model,to_file='model.png')


import os
path="/Users/erdemkok/Desktop/AITest/images/"
path2="/Users/erdemkok/Desktop/AITest/areas/"
nesneler = os.listdir(path)
nesneler2 = os.listdir(path2)
print("Nesneler ->",nesneler)
NesneImages2 = []
NesneImages = []
#os.remove("/Users/erdemkok/Desktop/AITest/areas/.DS_Store")
for img_isim in os.listdir(path):
    
    img_url = path+img_isim
    print(img_url)
    
    
    img = cv2.imread(img_url)
    img1=cv2.resize(img,(200,200))
    NesneImages2.append(img1)
    #img = cv2.resize(img,(20,20))
    # #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # img = img.reshape(50,50,3)
    # # img = img/255
    #NesneImages.append(img)
for img_isim in os.listdir(path2):
    img_url = path+img_isim
    print(img_url)
    
    
    img = cv2.imread(img_url)
    img = cv2.resize(img,(20,20))
    # #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # img = img.reshape(50,50,3)
    # # img = img/255
    NesneImages.append(img)
    
print("BURSa",len(NesneImages),len(NesneImages2))
NesneImages=np.array(NesneImages)
NesneImages2=np.array(NesneImages2)
print(type(NesneImages2),type(NesneImages2[0]))
# X,Y = veriDiz2(NesneImages)
# x2,y2=veriDiz2(NesneImages2)

# print(X.shape)


# """Train test split function for train and predict works
# """

# Xana = shuffle(X[0],random_state=0)
# Xkarsit = shuffle(X[1],random_state=0)
# Y = shuffle(Y,random_state=0) 

# XanaTrain = Xana[:(len(Xana)//10)*9]
# XkarsitTrain = Xkarsit[:(len(Xkarsit)//10)*9]
# YTrain = Y[:(len(Y)//10)*9]

# XanaTest = Xana[(len(Xana)//10)*9:]
# XkarsitTest= Xkarsit[(len(Xkarsit)//10)*9:]
# YTest = Y[(len(Y)//10)*9:]

X=[]
x2=[]
a=cv2.imread("/Users/erdemkok/Desktop/AITest/images/1.jpeg")

a=cv2.resize(a,(200,200))
X.append(a)
b=cv2.resize(a,(20,20))
x2.append(b)
X=np.array(X)
x2=np.array(x2)
ar1=[]
ar=np.array([0.5,0.3,0.6,0.2])
ar1.append(ar)
ar1=np.array(ar1)
print(type(X),type(ar),ar)
num_samples = 1000
image_640x480 = np.random.random((num_samples, 200, 200, 3))
image_50x50 = np.random.random((num_samples, 20, 20, 3))
input_values = np.random.random((55, 4))
labels = np.random.random((55, 4))
print(labels[0],type(image_50x50[0]))
# cv2.imshow("1",image_640x480[0])
# cv2.waitKey(0)
# print(image_640x480),
#model.fit(x =[X, x2, ar1],y =ar1,batch_size=32,epochs=50,verbose=1)
# import sys
# sys.exit()
#model.fit(x =[image_640x480, image_50x50, input_values],y =labels,batch_size=32,epochs=50,verbose=1)

inputt=[]
file_path = '/Users/erdemkok/Desktop/AITest/readme.txt'  # Replace with the actual file path

with open(file_path, 'r') as file:
    for line in file:
        #line = line.strip()  # Remove leading/trailing whitespaces and newline characters
        # Process the line
        # ...

        # Example: Print each line
        #line = np.fromstring(line, dtype=float, sep=' ')
        line=np.array(eval(line))
        inputt.append(line)
inputt=np.array(inputt)
print(inputt,type(inputt[0]))
import sys
#sys.exit()
NesneImages2 = shuffle(NesneImages2,random_state=0)
NesneImages = shuffle(NesneImages,random_state=0)
inputt = shuffle(inputt,random_state=0) 

# model.fit(x =[NesneImages2, NesneImages, inputt],y =inputt,batch_size=32,epochs=200,verbose=0)
# model.save("Test.h5")

print(model.summary())

new_model = tf.keras.models.load_model('Test.h5')
print((np.expand_dims(inputt[0],axis=0)).shape)
print(new_model.predict([NesneImages2[5].reshape(1,200,200,3),NesneImages[5].reshape(1,20,20,3),np.expand_dims(inputt[5],axis=0)]))
cv2.imshow("NES",NesneImages2[5])
cv2.imshow("NES2",NesneImages[5])
cv2.waitKey(0)
