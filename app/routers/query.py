from fastapi import APIRouter
from app.services.pinecone import PineconeService


router = APIRouter()

pinconeServices = PineconeService()

@router.get("/query/")
async def query(query: str):
    return pinconeServices.hybrid_query(query)

@router.get("/query/{query}")
async def read_user(query: str):
    return pinconeServices.query()


