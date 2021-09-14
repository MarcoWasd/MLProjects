# Dataset pulled from Keggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

batch = 16


train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_datagen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1,
    rescale=1/255.,
    rotation_range=0.2)

train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\marco\\Desktop\\python\\chest_xray\\chest_xray\\train',
        batch_size=batch,
        target_size=(250, 250),
        class_mode='categorical',
        color_mode="grayscale")

validation_generator = val_datagen.flow_from_directory(
        'C:\\Users\\marco\\Desktop\\python\\chest_xray\\chest_xray\\val',
        batch_size=batch,
        target_size=(250, 250),
        class_mode='categorical',
        color_mode="grayscale")

test_generator = test_datagen.flow_from_directory(
        'C:\\Users\\marco\\Desktop\\python\\chest_xray\\chest_xray\\test',
        batch_size=batch,
        target_size=(250, 250),
        class_mode='categorical',
        color_mode="grayscale")

model = keras.models.Sequential()

model.add(Conv2D(filters=64, kernel_size=(7,7), padding='same', activation='relu', input_shape=(250, 250, 1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=512, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(train_generator, validation_data = validation_generator, epochs=50)

# Accuracy of 0.97 after 50 epochs



















