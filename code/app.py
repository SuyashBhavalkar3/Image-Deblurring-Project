import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import torch
from restormer import Restorer  # Ensure you have the correct import for Restormer

# Function to load the Restormer model
def load_restormer_model():
    model = Restorer()  # Initialize your Restormer model here
    model.load_state_dict(torch.load("path_to_your_restormer_model.pth"))  # Load the model weights
    model.eval()
    return model

# Function to apply a sharpening filter
def deblur_image_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function to apply unsharp masking
def deblur_image_unsharp(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# Function to apply Restormer
def deblur_image_restormer(image, model):
    # Convert the image to tensor format if necessary
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # Convert to [1, C, H, W]
    with torch.no_grad():
        deblurred_tensor = model(image_tensor)
    deblurred_image = deblurred_tensor.squeeze().permute(1, 2, 0).numpy()  # Convert back to [H, W, C]
    deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)  # Ensure pixel values are valid
    return deblurred_image

# Function to check blur using combined metrics (placeholder)
def combined_blur_detection(image, thresholds):
    # Replace this with actual blur detection code
    return False, {'Metric1': 1.0, 'Metric2': 2.0}  # Placeholder metrics

# Streamlit application
st.title("Image Deblurring Application")
st.write("Upload an image to deblur it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.subheader("🖼 Original Image")
    st.image(image, use_column_width="always")

    deblurring_method = st.selectbox("Select Deblurring Method", ["Sharpening Filter", "Unsharp Masking", "Restormer"])
    
    # Initialize deblurred_image_pil
    deblurred_image_pil = image.copy()  # Default to original image in case of failure

    # Always apply the selected deblurring method
    if deblurring_method == "Sharpening Filter":
        deblurred_cv = deblur_image_sharpening(image_cv)
    elif deblurring_method == "Unsharp Masking":
        deblurred_cv = deblur_image_unsharp(image_cv)
    elif deblurring_method == "Restormer":
        model = load_restormer_model()
        if model is None:
            st.error("⚠ Failed to load Restormer model.")
        else:
            deblurred_cv = deblur_image_restormer(image_cv, model)
    else:
        st.error("⚠ Unsupported deblurring method selected.")

    # Convert deblurred image to RGB for display
    deblurred_rgb = cv2.cvtColor(deblurred_cv, cv2.COLOR_BGR2RGB)
    deblurred_image = Image.fromarray(deblurred_rgb)

    st.subheader("🖼 Deblurred Image")
    st.image(deblurred_image, use_column_width="always")

    # Convert deblurred image to bytes for download
    buf = io.BytesIO()
    deblurred_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Deblurred Image",
        data=byte_im,
        file_name="deblurred_image.png",
        mime="image/png"
    )
