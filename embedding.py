import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import os

# 配置
MODEL_PATH = "/home/pangu/gxa_main/LLM_Model/nomic_v1.5/nomic-embed-text-v1.5" # ⚠️请确保这里是 config.json 所在的真实路径
PORT = 8999
DEVICE = "cuda:4" # 对应你的 CUDA_VISIBLE_DEVICES=4

app = FastAPI()

print(f"正在加载模型: {MODEL_PATH} 到 {DEVICE} ...")
# Nomic v1.5 需要 trust_remote_code=True
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device=DEVICE)
print("模型加载完成！")

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "nomic-embed-text-v1.5"

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    sentences = request.input
    if isinstance(sentences, str):
        sentences = [sentences]
    
    # 计算 Embeddings
    embeddings = model.encode(sentences, normalize_embeddings=True)
    
    data = []
    for i, emb in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": emb.tolist()
        })
    
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(len(s) for s in sentences), # 简单估算
            "total_tokens": sum(len(s) for s in sentences)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)