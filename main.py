from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import requests
import socket
import os

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_API_TOKEN2 = os.environ['HF_API_TOKEN']
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbedRequest(BaseModel):
    texts: list[str]


@app.get("/")
async def root():
    return {"message": "API Running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hf_token_exists": bool(HF_API_TOKEN)
    }


# --------------------------------------------------
# DNS TEST
# --------------------------------------------------
@app.get("/dns-test")
async def dns_test():
    try:
        ip = socket.gethostbyname("api-inference.huggingface.co")

        return {
            "status": "success",
            "hostname": "api-inference.huggingface.co",
            "ip": ip
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


# --------------------------------------------------
# INTERNET TEST
# --------------------------------------------------
@app.get("/internet-test")
async def internet_test():
    try:
        response = requests.get(
            "https://www.google.com",
            timeout=10
        )

        return {
            "status": "success",
            "http_status": response.status_code
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


# --------------------------------------------------
# HUGGING FACE TEST
# --------------------------------------------------
@app.get("/hf-test")
async def hf_test():
    try:
        response = requests.get(
            "https://api-inference.huggingface.co",
            timeout=20
        )

        return {
            "status": "success",
            "http_status": response.status_code,
            "response": response.text[:200]
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }



@app.get("/hf-router-test")
def hf_router_test():
    try:
        API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"

        headers = {
            "Authorization": f"Bearer {os.environ['HF_API_TOKEN']}"
        }

        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": "Today is a sunny day and I will get some ice cream."
            },
            timeout=60
        )

        return {
            "status_code": response.status_code,
            "response": response.json()
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

@app.get("/debug-token")
def debug_token():
    return {
        "exists": bool(HF_API_TOKEN),
        "length": len(HF_API_TOKEN),
        "prefix": HF_API_TOKEN if HF_API_TOKEN else None,
        "prefix2": HF_API_TOKEN2 if HF_API_TOKEN2 else None
    }
# --------------------------------------------------
# EMBEDDING TEST
# --------------------------------------------------
@app.post("/embed")
async def embed(req: EmbedRequest):
    try:
        client = InferenceClient(
            token=HF_API_TOKEN
        )

        embeddings = client.feature_extraction(
            req.texts,
            model=EMBEDDING_MODEL
        )

        return {
            "status": "success",
            "count": len(embeddings),
            "dimension": len(embeddings[0]),
            "sample": embeddings[0][:5].tolist()
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

