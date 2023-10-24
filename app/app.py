import streamlit as st
import cv2
import os
from utils import load_image, calculate_histogram, extract_vgg_features, display_similar_images
from keras.applications.vgg16 import VGG16

st.set_page_config(
    page_title="Search Engine",
    page_icon="🔎",
    layout="wide")

st.title("Image Search Engine Application")

with st.sidebar:
    st.title("Search Engine")
    st.image(load_image("app/logo.png"), width=100)
    st.header("Parameters")
    k = st.slider('Select the number of images to display', min_value =1, max_value = 10, value = 3)
    feature_type = st.selectbox("Select feature type:", ["Color Histogram", "VGG-16"])
    distance_type = st.selectbox("Select a metric:", ["Euclidean Distance", "Manhattan Distance", "Cosine Similarity"])

database_images = []
for filename in os.listdir("database"):
    img = cv2.imread(os.path.join("database", filename))
    database_images.append((f"database/{filename}", img))

model = VGG16(weights='imagenet', include_top=False)

columns = st.columns(k)

for i, column in enumerate(columns):
    column.image(database_images[i][0], use_column_width=True)

selected_image = st.empty()

clicked_image_index = None
for i, column in enumerate(columns):
    if column.button(f"Select Image {i + 1}"):
        clicked_image_index = i

if clicked_image_index is not None:
    input_image = cv2.imread(database_images[clicked_image_index][0])
    input_image = cv2.resize(input_image, (200, 200))

    if feature_type == "Color Histogram":
        input_vector = calculate_histogram(input_image)
        database_vectors = [(path, calculate_histogram(image)) for path, image in database_images]

    if feature_type == "VGG-16":
        input_vector = extract_vgg_features(database_images[clicked_image_index][0], model)
        database_vectors = [(path, extract_vgg_features(path, model)) for path, image in database_images]
    
    display_similar_images(distance_type, input_vector, database_vectors)