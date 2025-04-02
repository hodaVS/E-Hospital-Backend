from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend import PrescriptionBackend
from typing import Optional
import json

app = FastAPI()
backend = PrescriptionBackend()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe_stream")
async def transcribe_stream(audio: UploadFile = File(...)):
    try:
        result, status_code = backend.process_transcription_request(audio)
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(text: str = Form(...)):
    try:
        result, status_code = backend.process_chat_request(text)
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_prescription")
async def save_prescription(prescription: str = Form(...)):
    try:
        prescription_data = json.loads(prescription)
        result, status_code = backend.save_prescription_data(prescription_data)
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=result.get("error"))
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)


