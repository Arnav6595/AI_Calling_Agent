import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy import create_engine, text
import pandas as pd

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
TABLE_NAMES = [
    "ongoing_projects_source",
    "upcoming_projects_source",
    "completed_projects_source",
    "historical_sales_source",
    "past_customers_source",
    "feedback_source"
]

# --- NEW: Intelligent Document Formatting ---
def format_document_from_row(doc) -> str:
    """Creates a clean, human-readable document from a Document object."""
    metadata = doc.metadata
    table_name = metadata.get('source_table', 'unknown')
    
    if "historical_sales" in table_name:
        return f"Historical Sale Record: Property {metadata.get('PropertyID', 'N/A')} was sold on {metadata.get('DateOfSale', 'N/A')} for INR {metadata.get('SalePrice(INR)', 'N/A')} by broker {metadata.get('BrokerID', 'N/A')}."
    elif "ongoing_projects" in table_name:
        return f"Ongoing Project Information: {metadata.get('ProjectName', 'N/A')} located at {metadata.get('Location', 'N/A')}. Current status is {metadata.get('CurrentStatus', 'N/A')} and is {metadata.get('PercentageCompletion', 'N/A')}% complete."
    elif "upcoming_projects" in table_name:
        return f"Upcoming Project Information: {metadata.get('ProjectName', 'N/A')} at {metadata.get('Location', 'N/A')}. Planned start date: {metadata.get('PlannedStartDate', 'N/A')}."
    elif "completed_projects" in table_name:
        return f"Completed Project Information: {metadata.get('ProjectName', 'N/A')} at {metadata.get('Location', 'N/A')}. Completed on {metadata.get('CompletionDate', 'N/A')}."
    elif "past_customers" in table_name:
        return f"Past Customer Information: {metadata.get('CustomerName', 'N/A')} (ID: {metadata.get('CustomerID', 'N/A')}) contacted via {metadata.get('ContactInfo', 'N/A')}."
    elif "feedback" in table_name:
        return f"Customer Feedback: {metadata.get('FeedbackText', 'N/A')} from customer ID {metadata.get('CustomerID', 'N/A')} on {metadata.get('FeedbackDate', 'N/A')} (Rating: {metadata.get('Rating', 'N/A')})."
    else:
        # Generic fallback
        text_parts = [f"{key.replace('_', ' ').title()}: {val}" for key, val in metadata.items() if key != 'source_table' and pd.notna(val) and val != 'None']
        return f"{table_name.replace('_', ' ').title()}: {'; '.join(text_parts)}"

def verify_collection_exists(connection_string: str, collection_name: str) -> bool:
    """Verify that the collection exists in NeonDB."""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM langchain_pg_collection WHERE name = :name"), {"name": collection_name}).fetchone()
            return result[0] > 0
    except Exception as e:
        logging.error(f"Error verifying collection: {e}")
        return False

def main():
    """Main function to test RAG retrieval from NeonDB."""
    if not NEON_DATABASE_URL:
        logging.error("NEON_DATABASE_URL not found in .env file.")
        return

    try:
        # 1. Verify collection exists
        logging.info(f"Verifying collection '{COLLECTION_NAME}' exists...")
        if not verify_collection_exists(NEON_DATABASE_URL, COLLECTION_NAME):
            logging.error(f"Collection '{COLLECTION_NAME}' not found in NeonDB. Run ingestion script first.")
            return

        # 2. Initialize the embedding model
        logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("Embedding model loaded.")

        # 3. Connect to PGVector store
        logging.info(f"Connecting to vector store '{COLLECTION_NAME}'...")
        vector_store = PGVector(
            connection_string=NEON_DATABASE_URL,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        logging.info("Successfully connected to the vector store.")
        
        logging.info("\n✅ RAG Retrieval Test is ready.")
        logging.info("Enter queries to retrieve raw context from the database.")
        logging.info(f"Available tables for filtering: {', '.join(TABLE_NAMES)}")

        # --- INTERACTIVE TEST LOOP ---
        while True:
            print("\n" + "="*50)
            user_query = input("Enter a query (or 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
            
            # Optional: Filter by table
            filter_table = input(f"Filter by table? ({', '.join(TABLE_NAMES)} or press Enter for none): ").strip()
            filter_dict = {"source_table": {"$eq": filter_table}} if filter_table in TABLE_NAMES else None
            
            k = input("Number of documents to retrieve (default 3): ").strip()
            k = int(k) if k.isdigit() else 3

            # 4. Retrieve relevant documents
            logging.info(f"Retrieving up to {k} documents for query: '{user_query}'")
            if filter_dict:
                logging.info(f"Applying filter: {filter_dict}")
            retrieved_docs_with_scores = vector_store.similarity_search_with_score(
                user_query, k=k, filter=filter_dict
            )
            
            print("\n--- Top Retrieved Documents ---")
            if not retrieved_docs_with_scores:
                print("No relevant documents found. Try refining your query or checking the vector store.")
                logging.warning("No documents retrieved. Ensure the query is relevant and the vector store is populated.")
                continue
            
            for i, (doc, score) in enumerate(retrieved_docs_with_scores):
                print(f"\n--- Document {i+1} (Score: {score:.4f}) ---")
                print(f"Formatted Content: {format_document_from_row(doc)}")
                print(f"Raw Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
            print("---------------------------------")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()