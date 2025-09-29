# ml/train_mobile_net.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATA_DIR = "ml/data"
IMG_SIZE = (224,224)
BATCH = 8

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=15, width_shift_range=0.1,
                                   height_shift_range=0.1, horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                              batch_size=BATCH, subset='training')
val_gen = train_datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                            batch_size=BATCH, subset='validation')

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=preds)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=6)

# Save model and class mapping
model.save("ml/model.h5")
import json
with open("ml/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved model and class_indices.json")
