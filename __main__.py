from fastapi import FastAPI
from setfit import SetFitModel
from pydantic import BaseModel
import uvicorn

# Cargar el modelo
model = SetFitModel.from_pretrained("sergifusterdura/dailynoteclassifier-setfit-v1.5-16-shot")

# Definir la aplicación FastAPI
app = FastAPI()

# Crear una clase para los datos de entrada
class TextInput(BaseModel):
    text: str

# Endpoint para la clasificación
@app.post("/classify/")
def classify(input_data: TextInput):
    text = input_data.text
    prediction = model(text)
    return {"text": text, "prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
