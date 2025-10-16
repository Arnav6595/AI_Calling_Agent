import os
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
vector_store = PGVector(
    collection_name="real_estate_embeddings",
    connection=create_engine(os.getenv("NEON_DATABASE_URL")),
    embeddings=embeddings,
    use_jsonb=True
)

results = vector_store.similarity_search("landmark project", k=5)
for doc in results:
    print(doc.page_content, doc.metadata)