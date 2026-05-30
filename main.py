from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbedRequest(BaseModel):
    texts: list[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/embed")
async def embed(req: EmbedRequest):
    try:
        client = InferenceClient(token=HF_API_TOKEN)
        result = client.feature_extraction(req.texts, model=EMBEDDING_MODEL)
        return {
            "status": "success",
            "count": len(result),
            "dim": len(result[0]),
            "sample": result[0][:5].tolist(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
