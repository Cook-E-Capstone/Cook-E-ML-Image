# Import dependencies
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image

# List of labels
class_labels = ['alpukat','anggur','apel','bayam','brokoli','durian','jagung','jahe','jambu','jeruk','kembang kol',
                'kiwi','kol','lengkuas','lobak','mangga','melon','mentimun','naga','nanas','paprika',
                'pisang','salak','semangka','singkong','stroberi','terong','tomat','ubi jalar','wortel']

# Load model
model = tf.keras.models.load_model('app/ml/ml_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})

# Load and preprocess the image
def predict(img):
    img = img.resize((224, 224))
    img = img.convert('RGB') 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, np.max(predictions[0]) # Yg np.max(predictions[0]) diround ke 2 angka belakang koma ye