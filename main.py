from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import keras_ocr
import matplotlib.pyplot as plt
import os

app = FastAPI()

# Загрузка модели OCR
my_model = keras.models.load_model('captcha_ocr_model.h5')
pipeline = keras_ocr.pipeline.Pipeline(detector=my_model)

@app.get('/')
def home():
    return HTMLResponse(content=open('index.html').read(), status_code=200)

@app.post('/process')
async def process(image: UploadFile = File(...)):
    # Получение загруженного изображения
    image_path = 'static/' + image.filename
    with open(image_path, 'wb') as f:
        f.write(image.file.read())

    # Обработка изображения с помощью OCR
    images = [keras_ocr.tools.read(image_path)]
    prediction_groups = pipeline.recognize(images)

    # Удаление временного изображения
    os.remove(image_path)

    # Возвращение результатов OCR в шаблон HTML
    return HTMLResponse(content=open('result.html').read(), status_code=200, 
                        media_type='text/html', context={'request': request, 'predictions': prediction_groups[0]})
