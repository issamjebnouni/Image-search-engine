import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
import streamlit as st


def load_image(image_path):
  im = Image.open(image_path)
  return np.array(im)

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def calculate_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog1 = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog1

def get_gabor_kernel(w, h,sigma_x, sigma_y, theta, fi, psi):
    "getting gabor kernel with those values"
    # Bounding box
    kernel_size_x = w
    kernel_size_y = h
    (y, x) = np.meshgrid(np.arange(0, kernel_size_y ), np.arange(0,kernel_size_x))
    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    #Calculate the gabor kernel according the formulae
    gb = np.exp(-1.0*(x_theta ** 2.0 / sigma_x ** 2.0 + y_theta ** 2.0 / sigma_y ** 2.0)) * np.cos(2 * np.pi * fi * x_theta + psi)
    return gb

def build_filters(w, h,num_theta, fi, sigma_x, sigma_y, psi):
    "Get set of filters for GABOR"
    filters = []
    for i in range(num_theta):
        theta = ((i+1)*1.0 / num_theta) * np.pi
        for f_var in fi:
            kernel = get_gabor_kernel(w, h,sigma_x, sigma_y, theta, f_var, psi)
            kernel = 2.0*kernel/kernel.sum()
            kernel = cv2.normalize(kernel, kernel, 1.0, 0, cv2.NORM_L2)
            filters.append(kernel)
    return filters

def extractFeatures(img):
    "A vector of 2n elements where n is the number of theta angles"
    "and 2 is the number of frequencies under consideration"
    filters =  build_filters(img.shape[0],img.shape[1],5,(0.75,1.5),2,1,np.pi/2.0)
    fft_filters = [np.fft.fft2(i) for i in filters]
    img_fft = np.fft.fft2(img)
    a =  img_fft * fft_filters
    s = [np.fft.ifft2(i) for i in a]
    k = [p.real for p in s]
    return k


def calculate_gabor(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Set parameters for Gabor filters
    w, h = img.shape[1], img.shape[0]
    num_theta = 5  # Number of orientations
    fi_values = (0.75, 1.5)  # Frequencies
    sigma_x, sigma_y = 2, 1  # Standard deviations for x and y
    psi = np.pi / 2.0  # Phase offset

    # Extract Gabor filters
    filters = build_filters(w, h, num_theta, fi_values, sigma_x, sigma_y, psi)

    # Extract features from the image using Gabor filters
    features = extractFeatures(img)
    return np.array(features).flatten()

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

