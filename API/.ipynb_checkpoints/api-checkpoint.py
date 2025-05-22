from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(r"C:\Users\Akila\potato diseasie\saved_models\1")
class_names = ['Early Blight','Late Blight','Healthy']

@app.get('/ping')
async def ping():
    return 'hello i am alive'

def read_file_img(data):
    image = np.array(Image.open(BytesIO(data)))
    return image 

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_img(await file.read())
    img_batches = np.expand_dims(image,0)
    predicted_class = class_names(np.argmax(model.predict(img_batches)))
    confidence = np.max(model.predict[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app,port=8005 ,host='localhost')