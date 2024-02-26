import keras
from keras.preprocessing.image import ImageDataGenerator

train_folder="./train/"
validation_folder="./validation/"


TRAIN_IMGS = 2370
VALIDATION_IMGS = 100

SIZE = (224,224)
BATCH_SIZE = 4
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    train_folder,
    target_size=SIZE,
    batch_size=BATCH_SIZE,
    class_mode='input',
    color_mode='grayscale'
)

validation_generator = datagen.flow_from_directory(
    validation_folder,
    target_size=SIZE,
    batch_size=BATCH_SIZE,
    class_mode='input',
    color_mode='grayscale'
)