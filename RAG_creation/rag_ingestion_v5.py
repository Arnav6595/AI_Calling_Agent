import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector  # Updated import

# --- SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress SQL debug logs to avoid "gibberish" in errors
logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
logging.getLogger('psycopg').setLevel(logging.WARNING)

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
BATCH_SIZE = 500  # Batch size for insertion

# Initialize embedding model globally
logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL}'")
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load embedding model: {e}")
    embeddings = None

def verify_table_schema(db_url: str) -> bool:
    """Verify that the langchain_pg_embedding table has the expected schema."""
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'langchain_pg_embedding';
                """)
            ).fetchall()
            columns = {row[0]: row[1] for row in result}
            logging.info(f"langchain_pg_embedding schema: {columns}")
            
            if 'cmetadata' not in columns:
                logging.error("Column 'cmetadata' not found in langchain_pg_embedding table.")
                return False
            if columns['cmetadata'] != 'jsonb':
                logging.warning("Column 'cmetadata' is not JSONB. Run migration: ALTER TABLE langchain_pg_embedding ALTER COLUMN cmetadata TYPE JSONB USING (cmetadata::JSONB);")
            if 'document' in columns and columns['document'] != 'text':
                logging.error("Column 'document' is not TEXT. Run migration: ALTER TABLE langchain_pg_embedding ALTER COLUMN document TYPE TEXT;")
                return False
            return True
    except Exception as e:
        logging.error(f"Error verifying table schema: {e}")
        return False

def load_and_prepare_documents_from_db(db_url: str, table_names: list) -> list:
    """Loads data from multiple tables, processes each separately, and prepares it for LangChain."""
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Smaller chunks
    logging.info("Connecting to database and loading data from source tables...")
    
    max_length = 0
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
                    
                    # Track max length
                    current_max = max(len(doc.page_content) for doc in split_docs)
                    if current_max > max_length:
                        max_length = current_max
                    logging.info(f"Max page_content length in '{table_name}': {current_max}")
                    
                except Exception as e:
                    logging.error(f"Could not load or process table '{table_name}'. Skipping. Error: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return []

    logging.info(f"Total document chunks to embed: {len(all_documents)}")
    logging.info(f"Overall max page_content length: {max_length}")
    return all_documents

def verify_collection_exists(db_url: str, collection_name: str) -> bool:
    """Verify that the collection exists in NeonDB."""
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM langchain_pg_collection WHERE name = :name"),
                {"name": collection_name}
            ).fetchone()
            return result[0] > 0
    except Exception as e:
        logging.error(f"Error verifying collection: {e}")
        return False

def main():
    """Main function to create and store vector embeddings in NeonDB."""
    if not NEON_DATABASE_URL:
        logging.error("NEON_DATABASE_URL not found in the .env file.")
        return

    if not embeddings:
        logging.error("Embedding model not initialized. Exiting.")
        return

    # Create SQLAlchemy engine for database connection
    try:
        engine = create_engine(NEON_DATABASE_URL)
        logging.info("SQLAlchemy engine created successfully.")
    except Exception as e:
        logging.error(f"Failed to create SQLAlchemy engine: {e}")
        return

    # Verify table schema
    if not verify_table_schema(NEON_DATABASE_URL):
        logging.error("Table schema verification failed. Exiting.")
        return

    # Load and process data
    documents = load_and_prepare_documents_from_db(NEON_DATABASE_URL, SOURCE_TABLE_NAMES)
    if not documents:
        logging.error("No valid documents to process. Exiting.")
        return

    # Verify collection and log existing state
    if verify_collection_exists(NEON_DATABASE_URL, COLLECTION_NAME):
        logging.info(f"Collection '{COLLECTION_NAME}' already exists. It will be deleted and recreated.")
    else:
        logging.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating new collection.")

    # Create vector store
    logging.info(f"Uploading {len(documents)} vectors to NeonDB in batches of {BATCH_SIZE}...")
    try:
        vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection=engine,
            embeddings=embeddings,  # Changed from embedding_function to embeddings
            use_jsonb=True
        )
        # Delete existing collection if needed
        vector_store.delete(collection_name=COLLECTION_NAME)

        # Add in batches
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            logging.info(f"Added batch {i // BATCH_SIZE + 1} ({len(batch)} documents)")

        logging.info("Success! Vector store created in NeonDB.")
        logging.info(f"Collection name: '{COLLECTION_NAME}'")
        logging.info("Each document includes:")
        logging.info("- Clean, NaN-free page_content (combined row text)")
        logging.info("- Original column values as metadata (stored as JSONB in cmetadata column)")
        logging.info("- 'source_table' metadata to filter by origin")
    except Exception as e:
        logging.error(f"Error during vector store creation: {e}")

if __name__ == "__main__":
    try:
        import psycopg
        logging.info("psycopg module loaded successfully.")
    except ImportError as e:
        logging.error(f"Failed to load psycopg: {e}. Please install 'psycopg[binary]' using: pip install psycopg[binary]")
        raise
    main()