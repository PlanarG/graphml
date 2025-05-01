import json
import logging
import uvicorn

import numpy as np
import asyncio
from typing import List
from pathlib import Path
from tqdm import tqdm
from fastapi import FastAPI


import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class Model:
    def __init__(self, embed_model: str):
        self.embed_model_name = embed_model

        self.model = AutoModel.from_pretrained(
            embed_model,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embedding(self, inputs: List[str], batch_size=128) -> Tensor:
        """Get the embedding for a list of inputs."""
        if len(inputs) > batch_size:
            results = []
            for i in tqdm(range(0, len(inputs), batch_size), desc="Getting embeddings"):
                results.append(self.get_embedding(inputs[i:i + batch_size], batch_size))
            return torch.cat(results, dim=0)
        
        batch_dict = self.tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

model = None
app = FastAPI()
lock = asyncio.Lock()

@app.post('/get_embedding')
async def get_embedding_endpoint(inputs: list[str]):
    async with lock:
        global model
        if model is None:
            model = Model(embed_model="intfloat/multilingual-e5-large-instruct")
        
        embedding = model.get_embedding(inputs)
        return json.dumps(embedding.cpu().numpy().tolist())

if __name__ == "__main__":
    uvicorn.run(app)
