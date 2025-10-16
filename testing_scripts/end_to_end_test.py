import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
import json

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- CONFIGURATION (Copied from main.py) ---
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "real_estate_embeddings"
EMBEDDING_MODEL = "hkunlp/instructor-large"
PLANNER_MODEL = "gpt-4o-mini"
ANSWERER_MODEL = "gpt-4o"
TABLE_DESCRIPTIONS = """
- "ongoing_projects_source": Details about projects currently under construction.
- "upcoming_projects_source": Information on future planned projects.
- "completed_projects_source": Facts about projects that are already finished.
- "historical_sales_source": Specific sales records, including price, date, and property ID.
- "past_customers_source": Information about previous customers.
- "feedback_source": Customer feedback and ratings for projects.
"""
# --- CORRECTED & ENHANCED PROMPTS ---
QUERY_FORMULATION_PROMPT = f"""
You are a query analysis agent. Your task is to transform a user's query into a precise search query for a vector database and determine the correct table to filter by.

**Available Tables:**
{TABLE_DESCRIPTIONS}

**User's Query:** "{{user_query}}"

**Your Task:**
1.  Rephrase the user's query into a clear, keyword-focused English question suitable for a database search.
2.  Identify the single most relevant table from the list above to find the answer.
3.  Respond ONLY with a JSON object containing "search_query" and "filter_table".

**Example:**
User's Query: "Property 001-A-120 ka price kya tha?"
Your Response:
{{
  "search_query": "sale price of property 001-A-120",
  "filter_table": "historical_sales_source"
}}
"""

ANSWER_SYSTEM_PROMPT = """
You are an expert AI assistant for a premier real estate developer.

## YOUR PERSONA
- You are professional, helpful, and highly knowledgeable. Your tone should be polite and articulate.

## CORE BUSINESS KNOWLEDGE (Use this for general questions)
- **Operational Cities:** We are currently operational in Mohali, Delhi NCR, and Chandigarh.
- **Property Types:** We offer luxury apartments, villas, and commercial properties.
- **Budget Range:** Our residential properties typically range from 60 lakhs to 15 crores.

## CORE RULES
1.  **Language Adaptation:** If the user's original query was in Hinglish, you MUST respond in natural, friendly Hinglish. If it was in English, respond in professional English.
2.  **Fact-Based Answers:** If you are provided with CONTEXT from a database lookup, you MUST prioritize that information to answer the user's question. Do not invent details. If the context is empty, you can use your Core Business Knowledge to provide a general but helpful response.
3.  **Stay on Topic:** Only answer questions related to real estate. If asked about other topics, politely steer the conversation back.
"""

# Initialize clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)

def run_full_test(audio_file_path: str):
    """Simulates the entire AI logic chain from an audio file."""
    logging.info(f"--- Running full test for: {audio_file_path} ---")
    
    # 1. STT (Whisper)
    logging.info("Step 1: Transcribing audio with Whisper...")
    with open(audio_file_path, "rb") as audio_file:
        user_text = client_openai.audio.transcriptions.create(model="whisper-1", file=audio_file).text
    logging.info(f"Whisper Transcription: '{user_text}'")

    # 2. Planner LLM
    logging.info("Step 2: Formulating search plan...")
    plan_response = client_openai.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[{"role": "user", "content": QUERY_FORMULATION_PROMPT.format(user_query=user_text)}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    plan = json.loads(plan_response.choices[0].message.content)
    logging.info(f"Search Plan: {plan}")

    # 3. RAG Retrieval
    logging.info("Step 3: Retrieving context from NeonDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PGVector(connection_string=NEON_DATABASE_URL, collection_name=COLLECTION_NAME, embedding_function=embeddings)
    
    search_query = plan.get("search_query", user_text)
    filter_table = plan.get("filter_table")
    search_filter = {"source_table": filter_table} if filter_table else {}
    
    retrieved_docs = vector_store.similarity_search(search_query, k=3, filter=search_filter)
    context_text = "\\n\\n".join([doc.page_content for doc in retrieved_docs])
    logging.info(f"Retrieved Context Snippet: {context_text[:200]}...")

    # 4. Answerer LLM
    logging.info("Step 4: Generating final answer...")
    final_prompt_messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "system", "content": f"Use the following CONTEXT to answer:\\n{context_text}"},
        {"role": "user", "content": f"My original question was: '{user_text}'"}
    ]
    final_response = client_openai.chat.completions.create(model=ANSWERER_MODEL, messages=final_prompt_messages)
    agent_response_text = final_response.choices[0].message.content
    
    print("\\n" + "="*50)
    print("✅ TEST COMPLETE")
    print(f"Final Text for TTS: {agent_response_text}")
    print("="*50)

if __name__ == "__main__":
    # Create a dummy audio file or use a real recording for testing
    sample_audio_path = "test_query.wav" 
    if not os.path.exists(sample_audio_path):
        print(f"Warning: Test audio file '{sample_audio_path}' not found. Please create one.")
    else:
        run_full_test(sample_audio_path)
