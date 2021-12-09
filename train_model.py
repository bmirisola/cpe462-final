import os.path
import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Just train model for one person at the moment. Add a class for each person.
# This should be automated in the future.
num_of_classes = 1
dataset = os.path.join(os.getcwd(),"data")
print(dataset)
validation_ratio = .15
batch_size = 4

train_ds = tf.keras.utils.image_dataset_from_directory(dataset,validation_split=validation_ratio, subset="training",
    seed=42,
    label_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

base_model = ResNet50(weights='imagenet', include_top = False)

for layer in base_model.layers:
    layer.trainable = False

global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(1, activation='sigmoid')(global_avg_pooling)

face_classifier = keras.models.Model(inputs=base_model.input, outputs=output, name='ResNet50')

face_classifier.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

face_classifier.save("model/face_classifier.h5")