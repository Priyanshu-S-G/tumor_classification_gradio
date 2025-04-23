import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("models/custom_cnn_model.keras")
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

def predict_tumor(img: Image.Image):
    img = img.resize((224, 224))              # match model input size
    img = img.convert("RGB")                  # ensure 3 channels
    img_array = np.array(img) / 255.0         # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    
    prediction = model.predict(img_array)[0]
    result = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return f"🧠 Predicted Tumor Type: {result} ({confidence:.2%} confidence)"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Brain Tumor Classifier")
    gr.Markdown("Upload an MRI image and click **Predict** to classify the tumor type.")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload MRI Image")
    
    predict_btn = gr.Button("🔍 Predict")
    output_text = gr.Textbox(label="Prediction")

    predict_btn.click(fn=predict_tumor, inputs=img_input, outputs=output_text)

demo.launch()
