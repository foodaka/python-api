from typing import Union

from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
from fastapi import UploadFile
from app.routers import query_router

from app.services import PineconeService


# from fastapi.responses import JSONResponse
from typing import List
import csv

app = FastAPI(
    title="fastapi",
    version=0.1,
    root_path="/"
)
pineconeOps = PineconeService()


app.include_router(query_router)


class Item(BaseModel):
    name: str
    description: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post('/upload/')
async def upload_file(file: UploadFile = Form(..., example="example.csv")):
    if file.filename.endswith('.csv'):
        content = await file.read()
        data = content.decode("utf-8").split("\n")
        csv_content = csv.reader(data)
        csv_list = list(csv_content)
        return {"filename": file.filename, "content": csv_list}
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")