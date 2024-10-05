from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoProcessor
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import io
import os
from fastapi.staticfiles import StaticFiles

# import sys
# import os

# Add the directory where colpali_engine is installed to the Python path
# sys.path.append('/Users/mqureshi/Documents/rag_system/myenv/lib/python3.12/site-packages')

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset,load_from_pdf

import google.generativeai as genai
import PIL.Image

from agent import SmartAgent

app = FastAPI()
# app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

class QueryRequest(BaseModel):
    query: str

# # Load your model and processor
model_name = "vidore/colpali"
model = ColPali.from_pretrained("vidore/colpaligemma-3b-mix-448-base", torch_dtype=torch.float16, device_map="mps").eval()
model.load_adapter(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# # Load saved embeddings
embeddings_path = "/tmp/embeddings.pt"

if os.path.exists(embeddings_path):
    ds = torch.load(embeddings_path)
else:
    ds = []

# # Endpoint to upload PDF and process it
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_path = "/tmp/uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(contents)
    
    images = load_from_pdf(pdf_path)

    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    global ds
    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

#     # Save embeddings to a file
    torch.save(ds, embeddings_path)
    return {"message": "PDF processed and embeddings saved."}

# # Endpoint to handle queries
@app.post("/query/")
async def query(request: QueryRequest):
    query_text = request.query
    queries = [query_text]

    # Load embeddings
    if os.path.exists(embeddings_path):
        ds = torch.load(embeddings_path)
    else:
        return {"message": "No embeddings found. Please upload a PDF first."}

    dataloader = DataLoader(
        queries,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_queries(processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
    )
    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)
    image_index = scores.argmax(axis=1)[0]

    # Load the image and save it temporarily
    images = load_from_pdf("/tmp/uploaded.pdf")
    result_image_path = f"/tmp/saved_image_{image_index}.png"
    images[image_index].save(result_image_path)

            # Read the saved image as bytes and convert it to a PIL Image
 

        # Ensure genai is correctly imported and used
    model_llm = genai.GenerativeModel("gemini-1.5-flash")


    genai.configure(api_key='AIzaSyC2ZyJC7RxLCZ7jz9zo-7OllmHjG1D2l64')

    # Pass the query and image to the LLM
    model_llm = genai.GenerativeModel("gemini-1.5-flash")
    with open(result_image_path, "rb") as img_file:
        image_bytes = img_file.read()

        image = Image.open(io.BytesIO(image_bytes))


    response = model_llm.generate_content([queries[0], image])

    return {"response": response.text}

# def process_query(queries, ds):
#     # Implement the logic to process the query using the embeddings
#     # For now, returning a placeholder response
#     return "This is a placeholder response."

# # New endpoint to handle agent queries
# @app.post("/agent_query/")
# async def agent_query(request: QueryRequest):
#     query_text = request.query
#     action = agent.decide_action(query_text)
#     response = agent.perform_action(action, query_text)
#     return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)