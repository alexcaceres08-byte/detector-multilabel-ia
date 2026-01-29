# app/main.py

# --- 1. CONFIGURACIÓN INICIAL PARA WINDOWS ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# ---------------------------------------------

# --- 2. IMPORTS (EL ORDEN ES CRÍTICO) ---
import shutil
import pandas as pd

# A. Importamos FastAI PRIMERO
# (Para que cargue sus cosas sin molestar a FastAPI después)
from fastai.vision.all import *

# B. Importamos FastAPI AL FINAL
# (Así aseguramos que File, Form y UploadFile sean los correctos)
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# --- 3. RUTAS Y DIRECTORIOS ---
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / 'data'
IMAGES_PATH = DATA_PATH / 'images'
CSV_PATH = DATA_PATH / 'labels.csv'

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar modelo
path_modelo = BASE_DIR / 'model.pkl'
try:
    learn = load_learner(path_modelo)
except Exception as e:
    print(f"Advertencia: No se pudo cargar el modelo ({e})")

# --- 4. FUNCIONES AUXILIARES ---
def obtener_imagen(fila):
    return IMAGES_PATH / fila['fname']

def obtener_etiquetas(fila):
    return fila['labels'].split(' ')

def reentrenar_modelo():
    print("--- Iniciando Re-entrenamiento ---")
    try:
        # A. Cargar datos
        df = pd.read_csv(CSV_PATH)
        df = df[df['fname'].apply(lambda x: (IMAGES_PATH/x).exists())]
        
        # B. Crear DataBlock
        dblock = DataBlock(
            blocks=(ImageBlock, MultiCategoryBlock),
            get_x=obtener_imagen,
            get_y=obtener_etiquetas,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            item_tfms=Resize(460),
            batch_tfms=aug_transforms(size=224, min_scale=0.75)
        )
        
        # C. DataLoaders (num_workers=0 es VITAL en Windows)
        dls = dblock.dataloaders(df, bs=32, num_workers=0)
        
        # D. Entrenar
        learn.dls = dls
        learn.fine_tune(1)
        
        # E. Guardar
        learn.export(path_modelo)
        return "¡Modelo actualizado con éxito!"
        
    except Exception as e:
        print(f"Error crítico: {e}")
        return f"Error: {e}"

# --- 5. RUTAS DEL SERVIDOR ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pred_class, pred_idx, probabilities = learn.predict(img_bytes)
    
    resultados = {}
    for clase, prob in zip(learn.dls.vocab, probabilities):
        resultados[clase] = float(prob)
    
    return {"confianza": resultados}

@app.post("/teach")
async def teach(
    file: UploadFile = File(...), 
    etiquetas: str = Form(...) 
):
    try:
        # 1. Guardar imagen
        filename = f"new_{file.filename}"
        save_folder = IMAGES_PATH / "user_added"
        save_folder.mkdir(exist_ok=True)
        
        file_location = save_folder / filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Guardar en CSV
        rel_path = f"user_added/{filename}"
        nueva_fila = pd.DataFrame({'fname': [rel_path], 'labels': [etiquetas]})
        nueva_fila.to_csv(CSV_PATH, mode='a', header=False, index=False)
        
        # 3. Re-entrenar
        mensaje = reentrenar_modelo()
        return {"status": "ok", "message": mensaje}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}