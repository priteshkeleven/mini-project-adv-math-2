import io
import wget
import os
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd


def load_model():
    loaded_model = tf.keras.models.load_model('fine_tuned_model_v1.h5')
    return loaded_model


def load_image():
    uploaded_file = st.file_uploader(label='Pick a food image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_labels():
    labels_file = os.path.basename('labels.txt')
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, image, categories):
    if image is not None:
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        with st.spinner('Wait for it...'):
            predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        st.metric('Confidence', f'{100 * np.max(score):.2f} %')
        st.metric('Prediction', f'{categories[np.argmax(score)]}')
        st.write(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(categories[np.argmax(score)], 100 * np.max(score))
        )
        top_5_pred = np.argsort(score)[-5:][::-1]
        top_5_pred_classes = [categories[i] for i in top_5_pred]
        top_5_scores = np.array(
            ['{:.2f} %'.format(100 * score[i]) for i in top_5_pred])
        st.subheader('Top 5 Predictions:')
        st.table(pd.DataFrame(
            {'Top 5 Predictions': top_5_pred_classes, 'Confidence': top_5_scores}))
    else:
        st.write('Please upload an image to test')


def main():
    st.title('Food101 Classification Demo (Fine-Tuned on ImageNetB0)')
    with st.expander("See All Classes"):
        st.write(', '.join(load_labels()))
    model = load_model()
    categories = load_labels()
    image = load_image()
    predict(model, image, categories)


if __name__ == '__main__':
    main()
