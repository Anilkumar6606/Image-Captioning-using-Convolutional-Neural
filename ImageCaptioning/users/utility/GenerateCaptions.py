import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from django.conf import settings
import os

# Set memory growth for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting error: {e}")

# Define the model architecture to match the saved model
def define_model(vocab_size, max_length):
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except Exception as e:
        return None, f"ERROR: Couldn't open image! {str(e)}"
    image = image.resize((299, 299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature, None


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    print("Caption generated (generate_desc called):", in_text)
    return in_text


def start_process(imagepath):
    max_length = 32
    model_path = os.path.join(settings.MEDIA_ROOT, "models", "model_9.h5")
    tokenizer_path = os.path.join(settings.MEDIA_ROOT, "models", "tokenizer.p")
    img_path = os.path.join(settings.MEDIA_ROOT, imagepath)
    if not os.path.exists(model_path):
        return None, "Model file not found. Please contact the administrator."
    if not os.path.exists(tokenizer_path):
        return None, "Tokenizer file not found. Please contact the administrator."
    if not os.path.exists(img_path):
        return None, "Image file not found."
    try:
        # Load tokenizer
        tokenizer = load(open(tokenizer_path, "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        
        # Create and load model weights
        model = define_model(vocab_size, max_length)
        model.load_weights(model_path)
        
        # Load Xception model
        xception_model = Xception(include_top=False, pooling="avg")
        print("Model and tokenizer loaded (start_process called).")
        
        photo, error = extract_features(img_path, xception_model)
        if error:
            return None, error
        description = generate_desc(model, tokenizer, photo, max_length)
        return description, None
    except Exception as e:
        print(f"Detailed error: {str(e)}")  # Add detailed error logging
        return None, f"Processing error: {str(e)}"


