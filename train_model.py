'''
All model code was based off tutorial at: https://gitlab.com/Winston-90/me_not_me_detector/-/tree/main
'''


import os.path
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import cv2

# Just train model for one person at the moment. Add a class for each person.
# This should be automated in the future.
num_of_classes = 1
dataset = os.path.join(os.getcwd(),"data")
validation_ratio = .15
batch_size = 16
img_path = "data/person/saved_img.jpg"
img = cv2.imread(img_path)
input_shape = img.shape
img_height = img.shape[0]
img_width = img.shape[1]
print(input_shape)

with tf.device('/CPU:0'):

    train_ds = tf.keras.utils.image_dataset_from_directory(dataset,validation_split=validation_ratio, subset="training",
        seed=42,
        image_size=(img_height,img_width),
        label_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        image_size=(img_height, img_width),
        validation_split=validation_ratio,
        subset="validation",
        seed=42,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True)

    base_model = ResNet50(weights='imagenet', include_top = False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(global_avg_pooling)

    face_classifier = keras.models.Model(inputs=base_model.input, outputs=output, name='ResNet50')

    # ModelCheckpoint to save model in case of interrupting the learning process
    checkpoint = ModelCheckpoint("model/face_classifier.h5",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    # EarlyStopping to find best model with a large number of epochs
    earlystop = EarlyStopping(monitor='val_loss',
                              restore_best_weights=True,
                              patience=4,  # number of epochs with no improvement after which training will be stopped
                              verbose=1)

    callbacks = [earlystop, checkpoint]

    face_classifier.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    history = face_classifier.fit(train_ds, callbacks=callbacks, epochs=50, validation_data=validation_ds)
    face_classifier.save("model/face_classifier.h5")