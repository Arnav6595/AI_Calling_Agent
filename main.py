import os
import base64
import logging
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.concurrency import run_in_threadpool  # Import for handling blocking calls
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine

# --- SETUP ---
# Suppress noisy logs from underlying libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SHARED_SECRET = os.getenv("SHARED_SECRET")

# --- CONFIGURATION ---
COLLECTION_NAME = "real_estate_embeddings"
EMBEDDING_MODEL = "hkunlp/instructor-large"
ELEVENLABS_VOICE_NAME = "Leo"
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

# --- GLOBAL VARIABLES FOR LIFESPAN ---
# These will be populated at startup
embeddings = None
vector_store = None

# --- FASTAPI LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    global embeddings, vector_store
    logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logging.info("Embedding model loaded successfully.")

    logging.info(f"Connecting to vector store '{COLLECTION_NAME}'...")
    engine = create_engine(NEON_DATABASE_URL, pool_pre_ping=True)
    vector_store = PGVector(
        connection=engine,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,  # <-- CRITICAL FIX: Corrected parameter name
    )
    logging.info("Successfully connected to the vector store.")
    yield
    # This code would run on shutdown (if needed)
    logging.info("Application shutting down.")

# --- INITIALIZE FastAPI APP WITH LIFESPAN ---
app = FastAPI(lifespan=lifespan)
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# --- PROMPTS ---
QUERY_FORMULATION_PROMPT = f"""
You are a query analysis agent. Your task is to transform a user's query into a precise search query for a vector database and determine the correct table to filter by.
**Available Tables:**
{TABLE_DESCRIPTIONS}
**User's Query:** "{{user_query}}"
**Your Task:**
1.  Rephrase the user's query into a clear, keyword-focused English question suitable for a database search.
2.  Identify the single most relevant table from the list above to find the answer.
3.  Respond ONLY with a JSON object containing "search_query" and "filter_table".
"""
ANSWER_SYSTEM_PROMPT = """
You are an expert AI assistant for a premier real estate developer.
## YOUR PERSONA
- You are professional, helpful, and highly knowledgeable. Your tone should be polite and articulate.
## CORE BUSINESS KNOWLEDGE
- **Operational Cities:** We are currently operational in Mohali, Delhi NCR, and Chandigarh.
- **Property Types:** We offer luxury apartments, villas, and commercial properties.
- **Budget Range:** Our residential properties typically range from 60 lakhs to 15 crores.
## CORE RULES
1.  **Language Adaptation:** If the user's original query was in Hinglish, respond in Hinglish. If in English, respond in English.
2.  **Fact-Based Answers:** Use the provided CONTEXT to answer the user's question. If the context is empty, use your Core Business Knowledge.
3.  **Stay on Topic:** Only answer questions related to real estate.
"""

# --- HELPER FUNCTIONS ---
def transcribe_audio(audio_bytes: bytes) -> str:
    """This is a blocking function."""
    try:
        with open("temp_audio.wav", "wb") as f: f.write(audio_bytes)
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = client_openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

async def formulate_search_plan(user_query: str) -> dict:
    logging.info("Formulating search plan with Planner LLM...")
    try:
        response = client_openai.chat.completions.create( # This can be async if using an async client
            model=PLANNER_MODEL,
            messages=[{"role": "user", "content": QUERY_FORMULATION_PROMPT.format(user_query=user_query)}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        plan = json.loads(response.choices[0].message.content)
        logging.info(f"Search plan received: {plan}")
        return plan
    except Exception as e:
        logging.error(f"Error in Planner LLM call: {e}")
        return {"search_query": user_query, "filter_table": None}

async def get_agent_response(user_text: str) -> str:
    """Runs the full RAG and generation logic for a given text query."""
    search_plan = await formulate_search_plan(user_text)
    search_query = search_plan.get("search_query", user_text)
    filter_table = search_plan.get("filter_table")

    search_filter = {"source_table": filter_table} if filter_table else {}
    if search_filter: logging.info(f"Applying filter: {search_filter}")

    retrieved_docs = vector_store.similarity_search(search_query, k=3, filter=search_filter)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    logging.info(f"Retrieved Context: {context_text[:500]}...")

    final_prompt_messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "system", "content": f"Use the following CONTEXT to answer:\n{context_text}"},
        {"role": "user", "content": f"My original question was: '{user_text}'"}
    ]
    final_response = client_openai.chat.completions.create(
        model=ANSWERER_MODEL,
        messages=final_prompt_messages
    )
    return final_response.choices[0].message.content

# --- API Endpoints ---
class TextQuery(BaseModel):
    query: str

@app.post("/test-text-query")
async def test_text_query_endpoint(query: TextQuery):
    """Endpoint for text-based testing via Swagger UI."""
    logging.info(f"Received text query: {query.query}")
    response_text = await get_agent_response(query.query)
    logging.info(f"Generated text response: {response_text}")
    return {"response": response_text}

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    auth_token = websocket.headers.get("x-auth-token")
    if not SHARED_SECRET or auth_token != SHARED_SECRET:
        logging.warning(f"Authentication failed. Closing connection.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logging.info("Authentication successful. Call connected.")
    try:
        while True:
            message = await websocket.receive_json()
            audio_base64 = message.get('audio')
            if not audio_base64: continue

            # PERFORMANCE FIX: Run blocking transcription in a separate thread
            user_text = await run_in_threadpool(
                transcribe_audio, base64.b64decode(audio_base64)
            )
            logging.info(f"User said: {user_text}")
            if not user_text.strip(): continue

            agent_response_text = await get_agent_response(user_text)
            logging.info(f"AI Responded: {agent_response_text}")

            # PERFORMANCE FIX: Run blocking audio generation in a separate thread
            audio_output = await run_in_threadpool(
                client_elevenlabs.generate,
                text=agent_response_text,
                voice=ELEVENLABS_VOICE_NAME,
                model="eleven_multilingual_v2"
            )
            response_audio_base64 = base64.b64encode(audio_output).decode('utf-8')
            await websocket.send_json({'audio': response_audio_base64})

    except WebSocketDisconnect:
        logging.info("Call disconnected.")
    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
    finally:
        await websocket.close()