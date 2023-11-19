import streamlit as st
import cv2
import os
from utils import load_image, calculate_histogram, calculate_hog, calculate_gabor, display_similar_images

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
    feature_type = st.selectbox("Select feature type:", ["RGB Histogram", "HOG Descriptor", "Gabor Descriptor"])
    distance_type = st.selectbox("Select a metric:", ["Euclidean Distance", "Manhattan Distance", "Cosine Similarity"])

database_images = []
for filename in os.listdir("database"):
    img = cv2.imread(os.path.join("database", filename))
    database_images.append((f"database/{filename}", img))

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

    if feature_type == "RGB Histogram":
        input_vector = calculate_histogram(input_image)
        database_vectors = [(path, calculate_histogram(image)) for path, image in database_images]

    if feature_type == "HOG Descriptor":
        input_vector = calculate_hog(input_image)
        database_vectors = [(path, calculate_hog(image)) for path, image in database_images]
    
    if feature_type == "Gabor Descriptor":
        input_vector = calculate_gabor(input_image)
        database_vectors = [(path, calculate_gabor(image)) for path, image in database_images]

    display_similar_images(distance_type, input_vector, database_vectors)