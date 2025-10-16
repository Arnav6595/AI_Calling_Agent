import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector  # Updated import
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text
import pandas as pd
from langdetect import detect

# --- SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
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
        return f"Ongoing Project Information: {metadata.get('ProjectName', 'N/A')} located at {metadata.get('Location', 'N/A')}. Current status is {metadata.get('CurrentStatus', 'N/A')} and is {metadata.get('PercentageCompletion', 'N/A')}% complete."
    elif "upcoming_projects" in table_name:
        return f"Upcoming Project Information: {metadata.get('ProjectName', 'N/A')} at {metadata.get('Location', 'N/A')}. Planned start date: {metadata.get('PlannedStartDate', 'N/A')}."
    elif "completed_projects" in table_name:
        return f"Completed Project Information: {metadata.get('ProjectName', 'N/A')} at {metadata.get('Location', 'N/A')}. Completed on {metadata.get('CompletionDate', 'N/A')}."
    elif "past_customers" in table_name:
        return f"Past Customer Information: {metadata.get('CustomerName', 'N/A')} (ID: {metadata.get('CustomerID', 'N/A')}) contacted via {metadata.get('ContactInfo', 'N/A')}."
    elif "feedback" in table_name:
        return f"Customer Feedback: {metadata.get('Comments', 'N/A')} from customer ID {metadata.get('CustomerID', 'N/A')} on {metadata.get('FeedbackDate', 'N/A')} (Rating: {metadata.get('Rating', 'N/A')})."  # Fixed FeedbackText to Comments
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

# --- COLLECTION VERIFICATION ---
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

# --- RAG RETRIEVAL AND GENERATION ---
def retrieve_and_generate(query: str, filter_table: str = None, k: int = 3) -> dict:
    """Retrieve documents from vector store and generate a response with GPT-4o."""
    try:
        # Verify environment variables
        if not NEON_DATABASE_URL:
            logging.error("NEON_DATABASE_URL not found in .env file.")
            return {"error": "Missing NEON_DATABASE_URL", "context": [], "response": "Sorry, an error occurred while processing your query."}

        if not OPENAI_API_KEY:
            logging.error("OPENAI_API_KEY not found in .env file.")
            return {"error": "Missing OPENAI_API_KEY", "context": [], "response": "Sorry, an error occurred while processing your query."}

        # Verify collection exists
        if not verify_collection_exists(NEON_DATABASE_URL, COLLECTION_NAME):
            logging.error(f"Collection '{COLLECTION_NAME}' not found in NeonDB.")
            return {"error": f"Collection '{COLLECTION_NAME}' not found", "context": [], "response": "Sorry, the required data is not available."}

        # Initialize embedding model
        logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("Embedding model loaded.")

        # Connect to PGVector store
        logging.info(f"Connecting to vector store '{COLLECTION_NAME}'...")
        engine = create_engine(NEON_DATABASE_URL)
        vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection=engine,  # Updated to use connection
            embeddings=embeddings,  # Updated to use embeddings
            use_jsonb=True
        )
        logging.info("Successfully connected to the vector store.")

        # Retrieve documents
        filter_dict = {"source_table": filter_table} if filter_table in TABLE_NAMES else None  # Simplified filter
        logging.info(f"Retrieving up to {k} documents for query: '{query}'")
        if filter_dict:
            logging.info(f"Applying filter: {filter_dict}")
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)

        if not retrieved_docs_with_scores:
            logging.warning("No documents retrieved.")
            return {
                "error": "No relevant documents found",
                "context": [],
                "response": "Sorry, I couldn't find any relevant information for your query. Please ask about our luxury apartments, villas, or commercial properties in Mohali, Delhi NCR, or Chandigarh."
            }

        # Format context for LLM
        context = [
            {"formatted": format_document_from_row(doc), "raw": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in retrieved_docs_with_scores
        ]
        context_text = "\n\n".join([doc["formatted"] for doc in context])

        # Detect language
        language = detect_language(query)
        logging.info(f"Detected language: {language}")

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        
        # Define prompt based on language
        if language == 'hinglish':
            prompt = ChatPromptTemplate.from_template(
                """
                Tum ek expert AI assistant ho jo ek premier real estate developer ke liye kaam karta hai. Tum professional, helpful, aur highly knowledgeable ho. Tum ek luxury brand ko represent karte ho, isliye tumhara tone polite aur articulate hona chahiye.

                **Rules**:
                - Agar context database se mila hai, to uss information ka use karke user ke sawal ka jawab do. Details invent mat karo.
                - Sirf real estate se related sawalon ka jawab do. Agar koi aur topic pe sawal ho, to politely conversation ko real estate ki taraf le jao.
                - Jawab natural aur friendly Hinglish mein do, jo phone call ke liye suitable ho.
                - Hamari company Mohali, Delhi NCR, aur Chandigarh mein operate karti hai, aur hum luxury apartments, villas, aur commercial properties offer karte hain (budget: 60 lakh se 15 crore).

                Context:
                {context}

                Query: {query}

                Response:
                """
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """
                You are an expert AI assistant for a premier real estate developer. You are professional, helpful, and highly knowledgeable. You represent a luxury brand, so your tone should be polite and articulate.

                **Rules**:
                - If provided with context from a database lookup, use that information to answer the user's question. Do not invent details.
                - Only answer questions related to real estate. If asked about other topics, politely steer the conversation back to real estate.
                - Provide a concise, professional response suitable for text-to-speech conversion in a phone call.
                - Our company operates in Mohali, Delhi NCR, and Chandigarh, offering luxury apartments, villas, and commercial properties (budget: 60 lakhs to 15 crores).

                Context:
                {context}

                Query: {query}

                Response:
                """
            )

        # Generate response
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "query": query}).content
        logging.info(f"Generated response: {response}")

        return {
            "error": None,
            "context": context,
            "response": response
        }

    except Exception as e:
        logging.error(f"Error during retrieval or generation: {e}")
        return {
            "error": str(e),
            "context": [],
            "response": "Sorry, an error occurred while processing your query. Please ask about our luxury properties in Mohali, Delhi NCR, or Chandigarh."
        }

def main():
    """Main function for testing RAG retrieval and LLM generation."""
    logging.info("Starting RAG retrieval and generation test...")
    logging.info(f"Available tables for filtering: {', '.join(TABLE_NAMES)}")

    # Example queries for testing
    test_queries = [
        {"query": "What was the SalePrice for Property 001-A-120", "filter_table": "historical_sales_source"},
        {"query": "Tell me about ongoing projects", "filter_table": "ongoing_projects_source"},
        {"query": "Show customer feedback", "filter_table": "feedback_source"},
        {"query": "Property ka price kya hai 001-A-120 ka?", "filter_table": "historical_sales_source"}
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
                print(f"Raw: {doc['raw']}")
                print(f"Metadata: {doc['metadata']}")
            print("\n--- LLM Response ---")
            print(result["response"])
            print("="*50)

if __name__ == "__main__":
    main()