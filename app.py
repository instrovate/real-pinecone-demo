
import streamlit as st
import pandas as pd
import openai
import pinecone
import os

# Set your API keys from Streamlit secrets or environment variables
openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"]
pinecone_index_name = st.secrets["PINECONE_INDEX"]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
if pinecone_index_name not in pinecone.list_indexes():
    st.error(f"Pinecone index '{pinecone_index_name}' not found.")
    st.stop()

index = pinecone.Index(pinecone_index_name)

# UI
st.title("📦 Pinecone RAG Demo – Microsoft Fabric Dataset")
st.markdown("Load sample data, embed it, and ask questions using GPT and Pinecone Vector DB.")

# Load CSV from GitHub
DATA_URL = "https://raw.githubusercontent.com/instrovate/real-pinecone-demo/main/Sample_Microsoft_Fabric_Dataset.csv"
df = pd.read_csv(DATA_URL)
texts = df["Text"].dropna().tolist()

# Step 1 – Embed & Upsert
if st.button("🔄 Embed & Upload to Pinecone"):
    with st.spinner("Embedding and uploading..."):
        vectors = []
        for i, text in enumerate(texts):
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response["data"][0]["embedding"]
            vectors.append((f"id-{i}", embedding, {"text": text}))
        index.upsert(vectors)
    st.success("✅ Data embedded and uploaded to Pinecone.")

# Step 2 – Query
query = st.text_input("🔍 Ask a question:")
if query:
    st.info("Searching...")
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]

    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    st.subheader("🔎 Top Matches")
    for match in result.get("matches", []):
        st.markdown(f"**Score**: {match['score']:.4f}")
        st.write(match["metadata"]["text"])
