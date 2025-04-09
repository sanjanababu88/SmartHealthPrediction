import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_DIR = "backend/models/skin_disease/dataset/"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")

datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, horizontal_flip=True)
train_generator = datagen.flow_from_directory(TRAIN_DIR, target_size=(299, 299), batch_size=32, class_mode="categorical")
valid_generator = datagen.flow_from_directory(VALID_DIR, target_size=(299, 299), batch_size=32, class_mode="categorical")

base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_generator, validation_data=valid_generator, epochs=10)
model.save("backend/models/skin_disease/saved_model/model.h5")
