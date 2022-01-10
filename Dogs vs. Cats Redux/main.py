# code was taken from existing publications on the
# Kaggle platform made by me
# link: https://www.kaggle.com/lewtas/cat-vs-dog-for-course?kernelSessionId=77380255
from glob import glob
import re
import zipfile
import shutil
import pandas as pd
from PIL import Image
import os
from glob import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

with zipfile.ZipFile("input/train.zip", "r") as z:
    z.extractall(".")

with zipfile.ZipFile("input/test.zip", "r") as z:
    z.extractall(".")

train = []                                     # create empty folder name as train
label = []                                     # create empty folder name as label
dct = {'cat': 0, 'dog': 1}
# os.listdir returns the lisкенкенкеt of files in the folder, in this case image class names
for i in os.listdir('./train'):

    train_class = i.split(".", 1)

    for j in train_class[0:1]:

        train.append(i)                         # append image file into the new folder train

        label.append(str(dct[j]))                     # append the name of folder as the label of the image file


full_df = pd.DataFrame({'Image': train, 'Labels': label})  # create data frame from dictionary with 2 coloums Image and Labels


b_size = 64
train_data_generator = ImageDataGenerator(
            rescale=1./255.,
            validation_split=0.2,
            horizontal_flip=True
            )


train_generator = train_data_generator.flow_from_dataframe(dataframe=full_df,
                                                           directory="./train/",
                                                           x_col="Image",
                                                           y_col="Labels",
                                                           subset="training",
                                                           batch_size=b_size,
                                                           shuffle=True,
                                                           color_mode="rgb",
                                                           target_size=(64, 64),
                                                           class_mode="categorical")


validation_generator = train_data_generator.flow_from_dataframe(dataframe=full_df,
                                                                directory="./train/",
                                                                x_col="Image",
                                                                y_col="Labels",
                                                                subset="validation",
                                                                batch_size=b_size,
                                                                shuffle=True,
                                                                color_mode="rgb",
                                                                target_size=(64, 64),
                                                                class_mode="categorical")
cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), padding='same',
                      input_shape=(64, 64, 3)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Conv2D(32, (3, 3)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(layers.Dropout(0.25))

cnn.add(layers.Conv2D(64, (3, 3), padding='same'))
cnn.add(layers.Activation('relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Conv2D(64, (3, 3)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(layers.Dropout(0.25))

cnn.add(layers.Flatten())
cnn.add(layers.Dense(256, kernel_regularizer=l2(0.01)))
cnn.add(layers.Activation('relu'))
cnn.add(layers.Dropout(0.3))
cnn.add(layers.Dense(2))
cnn.add(layers.Activation('softmax'))
#   categorical_crossentropy
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()


history = cnn.fit(train_generator,
                  epochs=20,
                  validation_data=validation_generator,
                  shuffle=True
                  )

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


plt.plot(history.history['val_accuracy'])
plt.title('model val_accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.show()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


plt.plot(history.history['val_loss'])
plt.title('model val_loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.show()


test_data_generator = ImageDataGenerator(rescale=1./255.)
test_generator = test_data_generator.flow_from_directory(directory='./test',
                                                         batch_size=b_size,
                                                         shuffle=False, color_mode='rgb',
                                                         target_size=(64, 64),
                                                         class_mode='categorical')

prediction = cnn.predict(test_generator)

classes = ['cat', 'dog']

prediction = np.argmax(prediction, axis=1)
submission = pd.DataFrame(columns=['id', 'label'], dtype=str)
submission['label'] = [str(i) for i in prediction]
submission['id'] = [(''.join(filter(str.isdigit, name))) for name in test_generator.filenames]

submission.sort_values(by=['id'])

submission.to_csv('my_submission.csv', index=False)
