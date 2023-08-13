import streamlit as st
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model

model2 = load_model("bags.h5")

def preprocess_image(image):
    new_width, new_height = 64, 64
    resized_image = cv2.resize(image, (new_width, new_height))
    return np.expand_dims(resized_image, axis=0)

def get_top_class_labels(predictions):
    new_labels = ["Garbage Bag", "Paper Bag", "Plastic Bag"]
    top_classes = np.argsort(predictions)[0][-3:][::-1]
    class_labels = [new_labels[i] for i in top_classes]
    probabilities = predictions[0][top_classes] * 100
    return class_labels, probabilities

def main():
    st.title("CNN Bag Classification")

    st.header("Problem Statement:")
    st.write("Developing a Multi-Class Image Classification and Segmentation Model for Bag Type Recognition in Diverse Environments")

    st.subheader("Overview:")
    st.write("In the context of various industries, including agriculture and recycling, thereis a growing need for accurate and automated recognition of different types of bags,such as plastic, paper, and garbage bags. To address this challenge, the goal of thisproject is to create a machine learning and Convolutional Neural Network (CNN) model that can accurately classify and segment bag images based on their material type. Thedataset provided contains synthetic images of plastic, paper, and garbage bags, along with annotations in COCO format.")

    st.subheader("Problem Description:")
    st.write("The project aims to tackle the problem of recognizing and categorizing bags into distinctclasses (plastic, paper, and garbage) while also providing segmentation masks for the regions corresponding to each bag in an image.")

    st.subheader("images Represented:")
    st.write("1.Garbage bags.")
    image_url1 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxR_tMFqQMxxgzt0KpRLOu8eBEZIVbrR9kCA&usqp=CAU"
    st.image(image_url1)
    st.write("2. Plastic bags.")
    image_url2 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT3J4HZiGUSEgZfTijHooriQisknl6hyxj4Kg&usqp=CAU"
    st.image(image_url2)
    st.write("3. Paper bags.")
    image_url3 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGIsPqLx1y53gRtm6i5Tu4XtduCQSkMRlngQ&usqp=CAU"
    st.image(image_url3)

    st.subheader("Upload your image to determine the class to which it belongs.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = uploaded_image.read()
        image_array = np.frombuffer(image, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        processed_image = preprocess_image(decoded_image)
        prediction = model2.predict(processed_image)

        class_labels, probabilities = get_top_class_labels(prediction)

        st.image(decoded_image, caption="Uploaded Image", use_column_width=True)

        for label, probability in zip(class_labels, probabilities):
            st.subheader(f"{label}: {probability:.2f}%")



    st.header("Disclaimer:")
    st.write("The image classification and segmentation predictions provided by our AI models are based onConvolutional Neural Networks (CNNs) trained on a synthetic dataset of plastic, paper, andgarbage bag images. While we have diligently developed and fine-tuned the models, it is crucial to acknowledge the following potential limitations:")
    st.header("1. Synthetic Nature of Dataset")
    # Paragraph under Header 30
    st.write("The training dataset comprises synthetic bag images generated from stock photos. These synthetic images may not fully capture the complexity, diversity, and variations present in real-world bag images, potentially affecting the models' ability to generalize to authentic scenarios.")

    # Header 31
    st.subheader("2. Segmentation Accuracy")
    # Paragraph under Header 31
    st.write("The segmentation masks provided by the models are based on annotations in the COCO format. However, the accuracy of segmentation may vary based on the quality of annotations and the models' capacity to detect fine-grained details.")

    # Header 32
    st.subheader("3. Limited Bag Types")
    # Paragraph under Header 32
    st.write("Our models are designed to classify and segment three specific bag types: plastic, paper, and garbage. They may not accurately recognize or segment other types of bags, materials, or classes.")

    # Header 33
    st.subheader("4. Environmental Factors")
    # Paragraph under Header 33
    st.write("The models' performance might be influenced by variations in lighting conditions, backgrounds, and bag orientations encountered in real-world settings, which may not have been adequately represented in the training data.")

    # Header 34
    st.subheader("5. Generalization to Real Environments")
    # Paragraph under Header 34
    st.write("While the models demonstrate proficiency on synthetic images, their ability to handle the intricacies of real-world environments, unforeseen scenarios, and challenging backgrounds may be constrained.")

    # Header 35
    st.subheader("6. No 100% Accuracy Guarantee")
    # Paragraph under Header 35
    st.write("Achieving absolute accuracy in bag type recognition and segmentation is challenging due to the complexity and variability of images. Our models may not achieve perfection in all cases.")

    # Header 36
    st.subheader("7. Human Expertise")
    # Paragraph under Header 36
    st.write("The predictions and segmentation masks should be interpreted alongside human expertise and judgment. Domain experts should validate the results before making critical decisions.")

    # Closing paragraph
    st.write("While we strive to provide valuable and informative predictions, it is essential to be aware of the inherent limitations stemming from the synthetic dataset and the models' architecture."
            " Users should consider the predictions as supportive insights rather than definitive outcomes. We welcome your feedback and understanding as we continue to enhance and refine our AI system."
            " Thank you for utilizing our service!")

if __name__ == "__main__":
    main()
