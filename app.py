import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
import os

# ✅ Load environment variables from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_ENV"] = st.secrets["PINECONE_ENV"]
os.environ["PINECONE_INDEX"] = st.secrets["PINECONE_INDEX"]

# ✅ Set OpenAI key
openai.api_key = os.environ["OPENAI_API_KEY"]

# ✅ Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX"]

# ✅ Create Pinecone index if it doesn't exist (1536-dim for text-embedding-ada-002)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ✅ Connect to the index
index = pc.Index(index_name)

# ✅ UI Setup
st.set_page_config(page_title="Pinecone RAG Demo", layout="centered")
st.title("📦 Pinecone RAG Demo – Microsoft Fabric Dataset")
st.markdown("Load sample data, embed it, and ask questions using GPT and Pinecone Vector DB.")

# ✅ Load sample CSV
DATA_URL = "https://raw.githubusercontent.com/instrovate/real-pinecone-demo/main/Sample_Microsoft_Fabric_Dataset.csv"
df = pd.read_csv(DATA_URL)
st.subheader("📄 Sample data loaded:")
st.write(df.head())

texts = df["text"].dropna().tolist()

# ✅ Step 1: Embed & Upload
if st.button("🔄 Embed & Upload to Pinecone"):
    with st.spinner("Embedding and uploading to Pinecone..."):
        vectors = []
        for i, text in enumerate(texts):
            response = openai.Embedding.create(
                input=[text],
                model="text-embedding-ada-002"
            )
            embedding = response["data"][0]["embedding"]
            vectors.append((f"id-{i}", embedding, {"text": text}))
        index.upsert(vectors)
    st.success("✅ Data embedded and uploaded to Pinecone.")

# ✅ Step 2: Query Interface
query = st.text_input("🔍 Ask a question about Microsoft Fabric:")
if query:
    with st.spinner("Generating results..."):
        response = openai.Embedding.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_embedding = response["data"][0]["embedding"]

        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        st.subheader("🔎 Top Semantic Matches")
        for match in result.get("matches", []):
            st.markdown(f"**Score**: {match['score']:.4f}")
            st.write(match["metadata"]["text"])
