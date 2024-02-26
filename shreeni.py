#digitalshreeni
# Encoder
#----------------------------------------------AMEYA--------------------------------------1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_gen import train_generator, validation_generator, SIZE, BATCH_SIZE,TRAIN_IMGS,VALIDATION_IMGS
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224,224, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
# Decoder
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse','accuracy'])
model.summary()
callback_save = ModelCheckpoint("/content/AnomalyDetector.h5",
									monitor="mean_squared_error")
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5)
def lr_schedule(epoch):
  initial_learning_rate = 0.1
  decay_factor = 0.1
  decay_steps = 10
  return initial_learning_rate * (decay_factor ** (epoch // decay_steps))

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

history = model.fit(
    train_generator,
    steps_per_epoch=TRAIN_IMGS//BATCH_SIZE,
    epochs=1000,
    validation_data=validation_generator,
    validation_steps=VALIDATION_IMGS//BATCH_SIZE,
    shuffle=True,
    callbacks=[callback_save]
)