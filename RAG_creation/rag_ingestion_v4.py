import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# --- SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

# --- CONFIGURATION ---
COLLECTION_NAME = "real_estate_embeddings"
EMBEDDING_MODEL = "hkunlp/instructor-large"
SOURCE_TABLE_NAMES = [
    "ongoing_projects_source",
    "upcoming_projects_source",
    "completed_projects_source",
    "historical_sales_source",
    "past_customers_source",
    "feedback_source"
]

def load_and_prepare_documents_from_db(db_url: str, table_names: list) -> list:
    """Loads data from multiple tables, processes each separately, and prepares it for LangChain."""
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    logging.info("Connecting to database and loading data from source tables...")
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            for table_name in table_names:
                try:
                    df = pd.read_sql_table(table_name, connection)
                    logging.info(f"Loaded {len(df)} rows from table '{table_name}'")
                    
                    if df.empty:
                        logging.warning(f"Table '{table_name}' is empty. Skipping.")
                        continue
                    
                    # Fill NaN/NaT with empty string
                    df = df.fillna('')
                    
                    # Create combined text column
                    df['combined_text'] = (
                        df.astype(str)
                        .agg(' '.join, axis=1)
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.strip()
                    )
                    
                    # Drop rows where combined_text is empty
                    initial_rows = len(df)
                    df = df[df['combined_text'] != '']
                    if len(df) < initial_rows:
                        logging.info(f"Dropped {initial_rows - len(df)} empty rows from '{table_name}'.")
                    
                    if df.empty:
                        logging.warning(f"No valid rows after cleaning in '{table_name}'. Skipping.")
                        continue
                    
                    # Create Document objects manually
                    table_documents = []
                    for _, row in df.iterrows():
                        metadata = {col: row[col] for col in df.columns if col != 'combined_text'}
                        metadata['source_table'] = table_name
                        doc = Document(
                            page_content=row['combined_text'],
                            metadata=metadata
                        )
                        table_documents.append(doc)
                    
                    # Split documents
                    split_docs = text_splitter.split_documents(table_documents)
                    all_documents.extend(split_docs)
                    logging.info(f"Created {len(split_docs)} document chunks from '{table_name}'.")
                    
                except Exception as e:
                    logging.error(f"Could not load or process table '{table_name}'. Skipping. Error: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return []

    logging.info(f"Total document chunks to embed: {len(all_documents)}")
    return all_documents

def main():
    """Main function to create and store vector embeddings in NeonDB."""
    if not NEON_DATABASE_URL:
        logging.error("NEON_DATABASE_URL not found in the .env file.")
        return

    # Load and process data
    documents = load_and_prepare_documents_from_db(NEON_DATABASE_URL, SOURCE_TABLE_NAMES)
    if not documents:
        logging.error("No valid documents to process. Exiting.")
        return

    # Initialize embedding model
    logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL}'")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        return

    # Create vector store
    logging.info(f"Uploading {len(documents)} vectors to NeonDB...")
    try:
        PGVector.from_documents(
            embedding=embeddings,
            documents=documents,
            collection_name=COLLECTION_NAME,
            connection_string=NEON_DATABASE_URL,
            pre_delete_collection=True
        )
        logging.info("Success! Vector store created in NeonDB.")
        logging.info(f"Collection name: '{COLLECTION_NAME}'")
        logging.info("Each document includes:")
        logging.info("- Clean, NaN-free page_content (combined row text)")
        logging.info("- Original column values as metadata")
        logging.info("- 'source_table' metadata to filter by origin")
    except Exception as e:
        logging.error(f"Error during vector store creation: {e}")

if __name__ == "__main__":
    main()