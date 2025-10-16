import os
import logging
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text
import pandas as pd
from langdetect import detect

# --- SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# --- DOCUMENT FORMATTING ---
def format_document_from_row(doc) -> str:
    """Creates a clean, human-readable document from a Document object."""
    metadata = doc.metadata
    table_name = metadata.get('source_table', 'unknown')
    
    if "historical_sales" in table_name:
        return f"Historical Sale Record: Property {metadata.get('PropertyID', 'N/A')} was sold on {metadata.get('DateOfSale', 'N/A')} for INR {metadata.get('SalePrice(INR)', 'N/A')} by broker {metadata.get('BrokerID', 'N/A')}."
    elif "ongoing_projects" in table_name:
        return f"Ongoing Project Information: {metadata.get('ProjectName', 'N/A')} at {metadata.get('Location', 'N/A')}. Status: {metadata.get('CurrentStatus', 'N/A')}, {metadata.get('PercentageCompletion', 'N/A')}% complete."
    # Add more specific formatters for other tables here if needed
    else:
        text_parts = [f"{key.replace('_', ' ').title()}: {val}" for key, val in metadata.items() if key != 'source_table' and pd.notna(val) and val != 'None']
        return f"{table_name.replace('_', ' ').title()}: {'; '.join(text_parts)}"

# --- LANGUAGE DETECTION ---
def detect_language(query: str) -> str:
    """Detect if query is in English or Hinglish based on content."""
    try:
        lang = detect(query)
        if lang == 'hi' or any(word in query.lower() for word in ['ghar', 'makan', 'property', 'flat', 'villa', 'crore', 'lakh']):
            return 'hinglish'
        return 'english'
    except Exception:
        return 'english'  # Default to English if detection fails

# --- RAG RETRIEVAL AND GENERATION ---
def retrieve_and_generate(query: str, filter_table: str = None, k: int = 3) -> dict:
    try:
        if not NEON_DATABASE_URL or not OPENAI_API_KEY:
            logging.error("API keys or DB URL not found in .env file.")
            return {"error": "Missing environment variables", "context": [], "response": ""}

        # 1. Initialize embedding model
        logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("Embedding model loaded.")

        # 2. Connect to PGVector store
        logging.info(f"Connecting to vector store '{COLLECTION_NAME}'...")
        vector_store = PGVector(
            connection_string=NEON_DATABASE_URL,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        logging.info("Successfully connected to the vector store.")

        # 3. Implement Hybrid Search Logic
        search_filter = {}
        property_id_match = re.search(r'\b\w{3}-\w-\d{3}\b', query)
        if property_id_match:
            found_id = property_id_match.group(0)
            logging.info(f"Property ID '{found_id}' detected. Applying exact match pre-filter.")
            search_filter = {"document": {"$like": f"%{found_id}%"}}
        
        if filter_table in TABLE_NAMES:
            table_filter = {"source_table": filter_table}
            search_filter.update(table_filter)
            logging.info(f"Applying table filter: {table_filter}")

        # 4. Retrieve documents
        logging.info(f"Retrieving up to {k} documents for query: '{query}'")
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(
            query, 
            k=k, 
            filter=search_filter
        )

        if not retrieved_docs_with_scores:
            logging.warning("No documents retrieved.")
            return {
                "error": "No relevant documents found",
                "context": [],
                "response": "Sorry, I couldn't find any information for that query."
            }

        # 5. Format context for LLM
        context_data = [
            {"formatted": format_document_from_row(doc), "raw": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in retrieved_docs_with_scores
        ]
        context_text = "\n\n".join([doc["formatted"] for doc in context_data])
        
        # 6. Generate response with LLM
        language = detect_language(query)
        logging.info(f"Detected language: {language}")

        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        
        prompt_template = ""
        if language == 'hinglish':
            prompt_template = """
            Tum ek expert AI assistant ho. Diye gaye context ka use karke user ke sawal ka jawab do.
            Context: {context}
            Query: {query}
            Response:
            """
        else:
            prompt_template = """
            You are an expert AI assistant. Use the provided context to answer the user's query.
            Context: {context}
            Query: {query}
            Response:
            """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        llm_response = chain.invoke({"context": context_text, "query": query}).content
        logging.info(f"Generated response: {llm_response}")

        return {
            "error": None,
            "context": context_data,
            "response": llm_response
        }

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return {"error": str(e), "context": [], "response": "An unexpected error occurred."}

# --- MAIN TEST LOOP ---
def main():
    """Main function for testing RAG retrieval and LLM generation."""
    logging.info("Starting RAG retrieval and generation test...")
    logging.info(f"Available tables for filtering: {', '.join(TABLE_NAMES)}")

    test_queries = [
        {"query": "What was the SalePrice for Property 001-A-120", "filter_table": "historical_sales_source"},
        {"query": "Tell me about ongoing projects", "filter_table": "ongoing_projects_source"},
    ]

    for test in test_queries:
        print("\n" + "="*50)
        logging.info(f"Testing query: '{test['query']}' with filter: {test['filter_table']}")
        result = retrieve_and_generate(test["query"], test['filter_table'], k=3)
        
        if result["error"]:
            print(f"Error: {result['error']}")
        else:
            print(f"Query: {test['query']}")
            print(f"Filter: {test['filter_table']}")
            print("\n--- Retrieved Context ---")
            for i, doc in enumerate(result["context"], 1):
                print(f"\nDocument {i} (Score: {doc['score']:.4f}):")
                print(f"Formatted: {doc['formatted']}")
            print("\n--- LLM Response ---")
            print(result["response"])
            print("="*50)

if __name__ == "__main__":
    main()

