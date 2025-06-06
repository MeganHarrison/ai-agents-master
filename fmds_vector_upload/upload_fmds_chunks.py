import os
import uuid
import openai
import pandas as pd
from supabase import create_client, Client
from tqdm import tqdm

# === CONFIGURATION ===
OPENAI_API_KEY = "your-openai-key"
SUPABASE_URL = "https://rpynhlesifkxqkokctmw.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # Truncated for safety
EMBEDDING_MODEL = "text-embedding-3-small"

# Load CSV or DataFrame of chunks
df = pd.read_csv("fmds_chunks.csv")  # Or load directly from prior code if in memory

# === SETUP ===
openai.api_key = OPENAI_API_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# === FUNCTION ===
def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print("Embedding error:", e)
        return None

# === PROCESS AND INSERT ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    embedding = get_embedding(row["text"])
    if not embedding:
        continue

    record = {
        "id": str(uuid.uuid4()),
        "section_number": row["section_number"],
        "section_title": row["section_title"],
        "text": row["text"],
        "page_number": int(row["start_page"]),
        "embedding": embedding
    }

    try:
        supabase.table("fmgs_chunks").insert(record).execute()
    except Exception as e:
        print("Upload error:", e)