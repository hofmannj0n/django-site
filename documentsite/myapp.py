import streamlit as st
from PIL import Image
import pytesseract

# Path to Tesseract within the virtual environment
tesseract_path_in_env = '/Users/jonathanhofmann/Desktop/documentsite/myenv/bin/tesseract'

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = tesseract_path_in_env

def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("OCR Key-Value Extractor")

    uploaded_file = st.file_uploader("Choose a document image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Text"):
            text = extract_text(image)
            st.subheader("Extracted Text:")
            st.text(text)

if __name__ == "__main__":
    main()

