# code was taken from existing publications on the
# Kaggle platform made by me
# link: https://www.kaggle.com/lewtas/cifar-10-for-kourse?kernelSessionId=77368974
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
import tensorflow as tf
import unarchive
import os
from PIL import Image
import pandas as pd
import shutil
import re
from glob import glob
#Imporm modules
#------------------------------------------------------------------------

# Load data


if(not os.path.exists('test')):
    unarchive_test()

if(not os.path.exists('train')):
    unarchive_train()

train_dir = os.listdir('train')
train_dir_len = len(train_dir)

train_labels = pd.read_csv('input/cifar-10/trainLabels.csv')
train_images = pd.DataFrame(columns=['id', 'label', 'path'], dtype=str)
test_labels = pd.read_csv('input/cifar-10/sampleSubmission.csv')
train_root = 'train/'
for i in range(0, train_dir_len):
    path = path = train_root + str(i+1) + '.png'
    if os.path.exists(path):
        train_images = train_images.append([{
            'id': train_labels['id'].iloc[i], 'label': train_labels['label'].iloc[i], 'path': str(i+1) + '.png'
        }])
size_img = 32
b_size = 64
quantity_classes = train_images['label'].unique().size
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for n in classes:
    index = classes.index(n)
    train_images.loc[train_images['label'] == n, 'label'] = str(index)

#------------------------------------------------------------------------

# Create generators for data normalize and comfortable work with them

data_generator = ImageDataGenerator(rescale=1/255.,
                                    validation_split=0.2,
                                    horizontal_flip=True)

train_generator = data_generator.flow_from_dataframe(dataframe=train_images,
                                                     directory='./train/',
                                                     x_col='path',
                                                     y_col='label',
                                                     subset='training',
                                                     batch_size=b_size,
                                                     shuffle=True,
                                                     target_size=(size_img, size_img),
                                                     class_mode='categorical')

validation_generator = data_generator.flow_from_dataframe(dataframe=train_images,
                                                          directory='./train/',
                                                          x_col='path',
                                                          y_col='label',
                                                          subset='validation',
                                                          batch_size=b_size,
                                                          shuffle=True,
                                                          target_size=(size_img, size_img),
                                                          class_mode='categorical')

#------------------------------------------------------------------------
# Create convertual neural network using tensorflow

cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), padding='same',
                      input_shape=(size_img, size_img, 3)))
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
cnn.add(layers.Dense(quantity_classes))
cnn.add(layers.Activation('softmax'))

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

#------------------------------------------------------------------------
# train the neural network

history = cnn.fit(train_generator,
                  epochs=50,
                  validation_data=validation_generator,
                  shuffle=True
                  )

#------------------------------------------------------------------------

# plotting graphs of changes accuracy of the neural network

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

#------------------------------------------------------------------------
# Create generator for test data and normalize test dataset

test_data_generator = ImageDataGenerator(rescale=1./255.)
test_generator = test_data_generator.flow_from_directory(directory='./test',
                                                         batch_size=b_size,
                                                         shuffle=False, color_mode='rgb',
                                                         target_size=(size_img, size_img),
                                                         class_mode=None)

#------------------------------------------------------------------------
# Predict on test data and save result on csv file

prediction = cnn.predict(test_generator)
prediction = np.argmax(prediction, axis=1)
submission = pd.DataFrame(columns=['id', 'label'], dtype=str)
submission['label'] = [classes[int(i)] for i in prediction]
submission['id'] = [(''.join(filter(str.isdigit, name))) for name in test_generator.filenames]

submission.sort_values(by=['id'])
submission.to_csv('my_submission.csv', index=False)
