import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class ImageToTextApp:
    def __init__(self):
        # Initialize the application and load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = self.load_model()

    @staticmethod
    @st.cache_resource
    def load_model():
        """
        Load the BLIP processor and model for image captioning.
        This function is cached to improve performance on subsequent runs.
        """
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Move the model to the appropriate device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return processor, model

    def generate_caption(self, image):
        """
        Generate a caption for a given image using the BLIP model.
        """
        # Preprocess the image and move it to the correct device
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Generate text from the image
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Capitalize the first letter of the caption
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        return caption

    def render_sidebar(self):
        """
        Render the sidebar with information about the app.
        """
        st.sidebar.title("About")
        st.sidebar.info(
            "This app uses the BLIP (Bootstrapping Language-Image Pretraining) model "
            "to generate captions for images."
        )

    def render_main(self):
        """
        Render the main interface of the app where users can upload images and get captions.
        """
        # Title and instructions for the app
        st.title("Image-to-Text Generation with BLIP")
        st.markdown(
            "Upload one or more images and click on 'Get Text' to generate descriptions."
        )

        # Allow users to upload multiple image files
        uploaded_files = st.file_uploader(
            "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )

        if uploaded_files:
            # Process and display each uploaded image
            images = [Image.open(file) for file in uploaded_files]
            for i, image in enumerate(images):
                st.image(image, caption=f"Uploaded Image {i+1}", use_container_width=True)

            # Button to generate captions for the uploaded images
            if st.button("Get Text"):
                with st.spinner("Generating captions..."):
                    # Generate captions for each image
                    captions = [self.generate_caption(image) for image in images]
                
                # Display captions in the order of the uploaded images
                for i, caption in enumerate(captions):
                    st.success(f"Caption for Image {i+1}:")
                    st.write(caption)

    def run(self):
        """
        Run the Streamlit application.
        """
        self.render_sidebar()  # Render the sidebar
        self.render_main()     # Render the main app interface

# Run the app
if __name__ == "__main__":
    app = ImageToTextApp()
    app.run()
