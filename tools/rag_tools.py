from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import os
import json

load_dotenv(find_dotenv())

vec_db_client = QdrantClient(url="http://localhost:6333", api_key="th3s3cr3tk3y")
collection_name = "architectures"

# Initialize the documents list
documents = []

# Initialize the model to None
model: HuggingFaceEmbedding = HuggingFaceEmbedding(
    model_name="llamaindex/vdr-2b-multi-v1",
    device="cpu",  # "mps" for mac, "cuda" for nvidia GPUs
    trust_remote_code=True,
)


def create_embeddings():
    """Useful for creating embeddings for images and text"""

    print("downloading the llamaindex/vdr-2b-multi-v1 model from huggingface")
    global model
    global documents

    # Specify the paths to the folders containing the images and descriptions
    images_folder = "images"
    descriptions_folder = "text"

    print("creating document structure")

    # Create a dictionary mapping image file names to their corresponding description files
    image_files = sorted(os.listdir(images_folder))
    description_files = sorted(os.listdir(descriptions_folder))

    # Generate the documents structure
    for image_file, description_file in zip(image_files, description_files):
        # Read the description content
        with open(os.path.join(descriptions_folder, description_file), "r") as f:
            description_content = f.read().strip()

        # Add the entry to the documents list
        documents.append({
            "architecture_description": description_content,
            "architecture_image": os.path.join(images_folder, image_file)
        })

    # Save the documents structure to a JSON file (optional)
    output_file = "documents.json"
    with open(output_file, "w") as f:
        json.dump(documents, f, indent=4)

    print("Generated documents structure:")
    print(json.dumps(documents, indent=4))


def ingest_to_vec_db():
    """Useful for ingesting the data to vector database"""
    print("starting ingestion...")

    text_embeddings = model.get_text_embedding_batch([doc["architecture_description"] for doc in documents],
                                                     show_progress=True)
    image_embeddings = []
    for doc in documents:
        image_embeddings.append(model.get_image_embedding(doc["architecture_image"]))

    print("creating collection in qdrant...")

    if not vec_db_client.collection_exists(collection_name=collection_name):
        vec_db_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(size=len(image_embeddings[0]), distance=models.Distance.COSINE),
                "text": models.VectorParams(size=len(text_embeddings[0]), distance=models.Distance.COSINE),
            }
        )

    print("inserting points into qdrant...")
    vec_db_client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector={
                    "text": text_embeddings[idx],
                    "image": image_embeddings[idx],
                },
                payload=doc
            )
            for idx, doc in enumerate(documents)
        ]
    )
    print("indexing completed.")


def retrieve_image_from_store(text_query: str):
    """
    Useful for retrieval of the data from vector database for a given text query
    :param text_query: (str) input query
    :return: None
    """
    print(f"started search...query: {text_query}")
    find_image = model.get_query_embedding(query=text_query)
    try:

        response = vec_db_client.query_points(
            collection_name=collection_name,
            query=find_image,
            using="image",
            with_payload=["architecture_image"],
            limit=1
        ).points[0].payload['architecture_image']

        print(response)
        image = Image.open(response)
        image.show()
    except Exception as e:
        print(e)


def retrieve_text_from_store(image_path: str):
    """
    Useful for retrieval of the data from vector database for a given image path
    :param image_path: (str) input image path to search
    :return:
    """
    response = vec_db_client.query_points(
        collection_name=collection_name,
        query=model.get_image_embedding(image_path),
        # Now we are searching only among text vectors with our image query
        using="text",
        with_payload=["architecture_description"],
        limit=1
    ).points[0].payload['architecture_description']

    print(response)
