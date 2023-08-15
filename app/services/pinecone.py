import pinecone
import os
import json
from dotenv import load_dotenv, find_dotenv

from sentence_transformers import SentenceTransformer
import torch

from pinecone_text.sparse import BM25Encoder
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import base64
from PIL import Image
from io import BytesIO

from datasets import load_dataset

import pickle
BM25_FILE = '../bm25_encoder.pkl'

# Get the absolute path to the directory the script is in
script_dir = os.path.dirname(os.path.abspath(__file__))


# Get the path to the images.pkl file
images_path = os.path.join(script_dir, 'images.pkl')

if os.path.exists(images_path):
    with open(images_path, "rb") as f:
        loaded_images = pickle.load(f)
else:
    print("Loading dataset and extracting metadata...")
    fashion = load_dataset(
        "ashraq/fashion-product-images-small",
        split="train"
    )

    # Assign the images and metadata to separate variables
    loaded_images = fashion["image"]
    metadata = fashion.remove_columns("image")
    metadata = metadata.to_pandas()

        # Save the images to images.pkl
    with open(images_path, "wb") as f:
        pickle.dump(loaded_images, f)


load_dotenv()

METADATA_FILE = 'metadata.pkl'

if os.path.exists(METADATA_FILE):
    # Load metadata from file
    print("Loading metadata from file...")
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
else:
    # Load dataset and extract metadata
    print("Loading dataset and extracting metadata...")
    fashion = load_dataset(
        "ashraq/fashion-product-images-small",
        split="train"
    )

    # Assign the images and metadata to separate variables
    images = fashion["image"]
    metadata = fashion.remove_columns("image")
    metadata = metadata.to_pandas()

    # Save the metadata to a file
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # load a CLIP model from huggingface
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device=device
)
# model


# Check if the serialized BM25 model exists
if os.path.exists(BM25_FILE):
    print("FILE EXISTS")
    # Load the serialized model
    with open(BM25_FILE, 'rb') as f:
        bm25 = pickle.load(f)
else:
    # Train the model from scratch
    print('WHY IN HERE')
    bm25 = BM25Encoder()
    bm25.fit(metadata['productDisplayName'])
    
    # Serialize (save) the trained model
    with open(BM25_FILE, 'wb') as f:
        pickle.dump(bm25, f)

class PineconeService:

    def __init__(self):
        _ = load_dotenv(find_dotenv())  # read local .env file
        api_key = os.getenv('PINECONE_KEY')
        api_env = os.getenv('PINECONE_ENVIRONMENT')

        pinecone.init(
            api_key=api_key,
            environment=api_env
        )
        self.index = None

    def create_index(self, index_name='default') -> list:
        # fetch the list of indexes
        indexes = pinecone.list_indexes()

        # create index if there are no indexes found
        if len(indexes) == 0:
            pinecone.create_index(index_name, dimension=8, metric="euclidean")

        return indexes

    def connect_index(self):
        indexes = self.create_index()
        # connect to a specific index
        self.index = pinecone.Index(indexes[0])

    def upsert(self, data):
        # sample data of the format
        # [
        #     ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        #     ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        #     ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        #     ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
        #     ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # ]
        # Upsert sample data (5 8-dimensional vectors)
        return json.loads(str(self.index.upsert(vectors=data, namespace="quickstart")).replace("'", '"'))

    def fetch_stats(self):
        # fetches stats about the index
        stats = self.index.describe_index_stats()
        return str(stats)

    def query(self, alpha=1.0):
        self.connect_index()

        query = "dark blue french connection jeans for men"

        # create sparse and dense vectors
        sparse = bm25.encode_queries(query)
        dense = model.encode(query).tolist()
        
        result = self.index.query(
            top_k=14,
            vector=dense,
            sparse_vector=sparse,
            include_metadata=True
        )
        print('result', result)
        
        return json.dumps(result.to_dict())

    def hybrid_query(self, query:str):
        self.connect_index()

        # create sparse and dense vectors
        sparse = bm25.encode_queries(query)
        dense = model.encode(query).tolist()

        hdense, hsparse = self.hybrid_scale(dense, sparse, alpha=1)
        result = self.index.query(
            top_k=14,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True
            )
         # Convert JpegImageFile to base64 and add to the result object
        for match in result["matches"]:
            match_id = int(match["id"])
            buffered = BytesIO()
            loaded_images[match_id].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            match["image"] = img_str
        return json.dumps(result.to_dict())


    def hybrid_scale(self, dense, sparse, alpha: float):
        """Hybrid vector scaling using a convex combination

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: float between 0 and 1 where 0 == sparse only
                and 1 == dense only
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse