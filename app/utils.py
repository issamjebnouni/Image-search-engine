import cv2
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
import streamlit as st

model = VGG16(weights='imagenet', include_top=False)

def load_image(image_path):
  im = Image.open(image_path)
  return np.array(im)

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_vgg_features(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def manhattan_distance(vector1, vector2):
  return np.sum(np.abs(vector1 - vector2))

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def display_similar_images(distance_type, input_vector, database_vectors):

  if distance_type == "Euclidean Distance":
    distances = [(path, euclidean_distance(input_vector, vector)) for path, vector in database_vectors]
    distances.sort(key=lambda x: x[1])
    top_similar_images = distances[:3]

    similar_images_columns = st.columns(3)
    for i, column in enumerate(similar_images_columns):
        img = load_image(top_similar_images[i][0])
        column.text(f"Euclidian Distance: {top_similar_images[i][1]: .2f}")
        column.image(img, use_column_width=True)

  if distance_type == "Manhattan Distance":
    distances = [(path, manhattan_distance(input_vector, vector)) for path, vector in database_vectors]
    distances.sort(key=lambda x: x[1])
    top_similar_images = distances[:3]

    similar_images_columns = st.columns(3)
    for i, column in enumerate(similar_images_columns):
        img = load_image(top_similar_images[i][0])
        column.text(f"Manhattan Distance: {top_similar_images[i][1]: .2f}")
        column.image(img, use_column_width=True)    

  if distance_type == "Cosine Similarity":
    distances = [(path, cosine_similarity(input_vector, vector)) for path, vector in database_vectors]
    distances.sort(key=lambda x: x[1], reverse= True)
    top_similar_images = distances[:3]

    similar_images_columns = st.columns(3)
    for i, column in enumerate(similar_images_columns):
        img = load_image(top_similar_images[i][0])
        column.text(f"Cosine Similarity: {top_similar_images[i][1]: .2f}")
        column.image(img, use_column_width=True)    

