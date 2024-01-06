import streamlit as st
import cv2
import numpy as np
from PIL import Image

def edge_detection(input_image, lower_threshold=50, upper_threshold=150, edge_thickness=1):
    try:
        # Convert PIL Image to NumPy array
        img = np.array(input_image)

        # Convert the image to grayscale using PIL
        gray_pil = Image.fromarray(img).convert('L')
        gray = np.array(gray_pil)

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

        # Create a blank image to draw the thickened edges
        thick_edges = np.zeros_like(img)

        # Draw the thickened edges on the blank image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick_edges = cv2.drawContours(thick_edges, contours, -1, (255, 255, 255), thickness=edge_thickness)

        return thick_edges

    except Exception as e:
        st.error(f"Error in edge_detection: {e}")
        return None

def main():
    st.title("Stamp Seal Maker with Edge Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Set Canny edge detection thresholds
        lower_threshold = st.slider("Lower Threshold", 0, 255, 50)
        upper_threshold = st.slider("Upper Threshold", 0, 255, 150)

        # Set thickness for the edges
        edge_thickness = st.slider("Edge Thickness", 1, 20, 1)

        if st.button("Generate Stamp Seal"):
            # Convert PIL Image to NumPy array
            pil_image = Image.open(uploaded_file)
            np_image = np.array(pil_image)

            # Perform edge detection with thickness adjustment
            thick_edges = edge_detection(np_image, lower_threshold, upper_threshold, edge_thickness)

            if thick_edges is not None:
                # Display the result
                st.image(thick_edges, caption="Edge Detection Result with Thickness.", use_column_width=True)

if __name__ == "__main__":
    main()
