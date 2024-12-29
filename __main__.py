from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # Cambia a pickle si usaste ese formato

# Cargar el modelo
model = joblib.load("modelo.pkl")  # Cambia el nombre del archivo según corresponda

# Definir la aplicación FastAPI
app = FastAPI()

# Crear una clase para los datos de entrada
class TextInput(BaseModel):
    text: str

# Endpoint para la clasificación
@app.post("/classify/")
def classify(input_data: TextInput):
    text = input_data.text
    prediction = model.predict([text])
    return {"text": text, "prediction": prediction[0]}
