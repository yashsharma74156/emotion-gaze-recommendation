import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ Dataset paths (corrected)
train_path = "../archive/train"
test_path = "../archive/test"

# ✅ Image dimensions
img_width, img_height = 48, 48
batch_size = 64
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral

# ✅ Image Data Generator with Augmentation
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ✅ CNN Model
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# ✅ Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Create models/ folder if it doesn't exist
if not os.path.exists("../models"):
    os.makedirs("../models")

# ✅ Save best model only
checkpoint = ModelCheckpoint("../models/emotion_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')

# ✅ Train
model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[checkpoint]
)
