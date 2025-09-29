import os
import cv2
import matplotlib.pyplot as plt  
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from sklearn.model_selection import train_test_split
from keras.models import load_model
# Define paths and columns
df = 'dataset5/'
cols = ['yes', 'no']
data1 = os.path.join(df, cols[0])
data2 = os.path.join(df, cols[1])

# List files in each directory
dt1 = os.listdir(data1)
dt2 = os.listdir(data2)

# Function to load and resize images
def loading(img_path):
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.resize(image, (69, 69))
        return image[..., ::-1]  # Convert BGR to RGB
    else:
        return None

# Create a figure with subplots
plt.figure(figsize=(12, 9))

# Determine the number of images to display
num_images_to_display = min(12, len(dt1), len(dt2))

# Nested for loop to iterate over files in both directories
for i in range(num_images_to_display):
    # Image from the 'yes' folder
    img_path1 = os.path.join(data1, dt1[i])
    plt.subplot(4, 6, i * 2 + 1)  # Adjust subplot indexing for display
    img1 = loading(img_path1)
    if img1 is not None:
        plt.imshow(img1)
    plt.title("yes")
    plt.axis('off')

    # Image from the 'no' folder
    img_path2 = os.path.join(data2, dt2[i])
    plt.subplot(4, 6, i * 2 + 2)  # Adjust subplot indexing for display
    img2 = loading(img_path2)
    if img2 is not None:
        plt.imshow(img2)
    plt.title("no")
    plt.axis('off')

plt.suptitle("Images from 'yes' and 'no' folders", fontsize=16)
plt.tight_layout()
plt.show()

data=[]
label=[]
for class_label in os.listdir(df):
    class_path=os.path.join(df,class_label)
    for img_file in os.listdir(class_path):
        img_path=os.path.join(class_path,img_file)
        img=load_img(img_path,target_size=(69,69))
        img_array=img_to_array(img)
        data.append(img_array)
        label.append(class_label)
x=np.array(data) 
y=np.array(label)


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_new=le.fit_transform(y)
y_new=to_categorical(y_new,num_classes=2)
print(y_new)

x_train,x_test,y_train,y_test=  train_test_split(x,y_new,test_size=0.2,random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten
model=Sequential()
model.add(Conv2D(32,(3,3), activation='relu',input_shape=(69,69,3)))#converted into pixel(dot multiplication)
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=20,batch_size=32,validation_data=(x_test,y_test))
model.save('model.h5')

plt.figure(figure=(12,9))
plt.title("accuracy graph")
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.figure(figure=(12,9))
plt.title("loss graph")
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


model=load_model("model.h5")
img="dataset5/yes/Y3.jpg"
img=load_img(img,target_size=(69,69))
array=img_to_array(img)
img_array=np.expand_dims(array,axis=0)
img_array/=255.0
prediction=model.predict(img_array)
predicted_class=np.argmax(prediction,axis=1)
pred=le.inverse_transform(predicted_class)
print(f"prediction is {pred[0]}")