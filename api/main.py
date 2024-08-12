from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
# from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

def combined_loss(y_true, y_pred):
    loss1 = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss2 = tf.keras.losses.categorical_hinge(y_true, y_pred)
    return 1.0 * loss1 + 0.0 * loss2

# Load the model
model_path = 'Wheat.h5'
model = load_model(model_path, custom_objects={'combined_loss':combined_loss})
def load_and_preprocess_image(img_path, image_size):
    
    img = Image.open(img_path).resize((image_size, image_size))
    img_array = np.array(img)
    img_array = img_array / 255.0  
    # print(img_array.shape)
    return img_array
# Define a route for predictions
@app.get('/')
def index():
    return {"data":"hello"}
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
  
    contents = await file.read()
    
    img_array = load_and_preprocess_image(io.BytesIO(contents), 224)
    img_array = img_array[np.newaxis, :]  
    # Make predictions
    predictions = model.predict(img_array)

    # Process the output and return it
    predicted_class = np.argmax(predictions,axis=1)[0]
    class_names = ['Wheat___Brown_Rust','Wheat___Healthy','Wheat___Yellow_Rust']
    return {"predicted_class": class_names[predicted_class]}
