#custom2
#==================================================================AMEYA2=============================================
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2DTranspose, BatchNormalization
from data_gen import train_generator, validation_generator, SIZE, BATCH_SIZE,TRAIN_IMGS,VALIDATION_IMGS
import os
seq = Sequential()
seq.add(Conv2D(512, (11, 11), strides=2, padding="same",input_shape=(224,224,1)))
seq.add(BatchNormalization())
seq.add(MaxPooling2D((2, 2), padding="same"))
seq.add(BatchNormalization())
seq.add(Conv2D(256, (5, 5), padding="same"))
seq.add(BatchNormalization())
seq.add(MaxPooling2D((2, 2), padding="same"))
seq.add(BatchNormalization())
seq.add(Conv2D(128, (3, 3), padding="same"))
seq.add(BatchNormalization())
# # # #
seq.add(Conv2DTranspose(128, (3, 3), padding="same"))
seq.add(BatchNormalization())
seq.add(UpSampling2D((2, 2)))
seq.add(BatchNormalization())
seq.add(Conv2DTranspose(256, (5, 5), padding="same"))
seq.add(BatchNormalization())
seq.add(UpSampling2D((2, 2)))
seq.add(BatchNormalization())
seq.add(Conv2DTranspose(512, (11, 11),strides=2, padding="same"))
seq.add(BatchNormalization())
seq.add(Conv2D(10, (11, 11), activation="sigmoid", padding="same"))
# seq.add(Dense(units=10,activation="sigmoid"))
seq.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(lr=1e-4,decay=1e-5,epsilon=1e-6), metrics=['mse','accuracy'])

callback_save = keras.callbacks.ModelCheckpoint("custom2.keras", monitor='val_loss', verbose=0, save_best_only=False,
                                            save_weights_only=False, mode='auto', period=5,)
BATCH_SIZE=64
seq.fit(train_generator,
    steps_per_epoch=2370//BATCH_SIZE,
    epochs=1000,
    validation_data=validation_generator,
    validation_steps=100//BATCH_SIZE,
    shuffle=True,
    callbacks=[callback_save])