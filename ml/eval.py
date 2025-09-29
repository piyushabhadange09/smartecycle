# ml/eval.py
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

model = tf.keras.models.load_model("ml/model.h5")
with open("ml/class_indices.json") as f:
    idx = json.load(f)
inv_idx = {v:k for k,v in idx.items()}

# Prepare test generator (use validation split)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_gen = datagen.flow_from_directory("ml/data", target_size=(224,224), batch_size=8, subset='validation', shuffle=False)

preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=[inv_idx[i] for i in range(len(inv_idx))]))

# plot confusion matrix
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = range(len(inv_idx))
plt.xticks(tick_marks, [inv_idx[i] for i in range(len(inv_idx))], rotation=45)
plt.yticks(tick_marks, [inv_idx[i] for i in range(len(inv_idx))])
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("ml/confusion_matrix.png")
print("Saved ml/confusion_matrix.png")
