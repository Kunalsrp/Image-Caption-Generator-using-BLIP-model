# Image-to-Text Generation with BLIP

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Result](#result)

## Overview

This project leverages the BLIP (Bootstrapping Language-Image Pretraining) model to generate captions for images. The application is built using Streamlit and supports multiple image uploads, providing a description for each image in the order they are uploaded.

## Features

- Upload multiple images and receive captions in sequence.

- Uses the BLIP model for generating accurate and context-aware captions.

- User-friendly interface with real-time processing.

## Technologies Used

- Streamlit: For creating the user interface.

- PyTorch: For running the BLIP model.

- Hugging Face Transformers: For utilizing the BLIP processor and model.

- Pillow: For image processing.

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/image-to-text-blip.git
```

### Navigate to the project directory
```bash
cd image-to-text-blip
```

### Create a virtual environment
```python
python -m venv venv
env/Scripts/activate   # Activate environment on Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit app

```
streamlit run app.py
```
- Open the Streamlit app in your browser (usually at http://localhost:8501).

- Upload one or more images.

- Click on "Get Text" to generate captions.

## Example

- Upload Images: Drag and drop multiple images or select them from your device.

- Generate Captions: Click the "Get Text" button to generate a caption for each image.

## Result

### **Uploading Images**
![Upload single or multiple images](image.png)

### **Generated Text for Image**
![Text Generated](Image2.png)