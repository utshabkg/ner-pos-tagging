import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'components')))
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from components.inference import ModelInference

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the ModelInference class
model_path = 'notebooks/models_evaluation/models/base_model.h5'
vocab_path = 'notebooks/analysis-preprocessing/processed_data/words_vocab.pkl'
pos_path = 'notebooks/analysis-preprocessing/processed_data/pos_vocab.pkl'
ner_path = 'notebooks/analysis-preprocessing/processed_data/ners_vocab.pkl'
model_inference = ModelInference(model_path, vocab_path, pos_path, ner_path)


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, sentence: str = Form(...)):
    pos_labels, ner_labels, tokens = model_inference.predict(sentence)
    results = [{"word": tokens[i], "pos": pos_labels[i], "ner": ner_labels[i]} for i in range(len(tokens))]
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "sentence": sentence})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
