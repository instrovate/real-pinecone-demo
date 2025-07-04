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

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# ✅ Create Pinecone index if it doesn't exist
index_name = os.environ["PINECONE_INDEX"]
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # adjust if using another region
        )
    )

# ✅ Connect to the index
index = pc.Index(index_name)

# ✅ Set OpenAI key
openai.api_key = os.environ["OPENAI_API_KEY"]

# ✅ Streamlit UI
st.set_page_config(page_title="Pinecone RAG Demo", layout="centered")
st.title("📦 Pinecone RAG Demo – Microsoft Fabric Dataset")
st.markdown("Load sample data, embed it, and ask questions using GPT and Pinecone Vector DB.")

# ✅ Load sample CSV
DATA_URL = "https://raw.githubusercontent.com/instrovate/real-pinecone-demo/main/Sample_Microsoft_Fabric_Dataset.csv"
df = pd.read_csv(DATA_URL)
st.write("📄 Sample data loaded:", df.head())

# ✅ Extract text from the correct column
texts = df["text"].dropna().tolist()

# ✅ Step 1 – Embed and Upload
if st.button("🔄 Embed & Upload to Pinecone"):
    with st.spinner("Embedding and uploading to Pinecone..."):
        vectors = []
        for i, text in enumerate(texts):
            response = openai.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            vectors.append((f"id-{i}", embedding, {"text": text}))  # ✅ store 'text' as metadata
        index.upsert(vectors)
    st.success("✅ Data embedded and uploaded to Pinecone.")

# ✅ Step 2 – Ask a Question
query = st.text_input("🔍 Ask a question about Microsoft Fabric:")
if query:
    with st.spinner("Generating results..."):
        try:
            query_embedding = openai.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            ).data[0].embedding

            result = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )

            st.subheader("🔎 Top Semantic Matches")
            for match in result.get("matches", []):
                score = match.get("score", 0)
                metadata = match.get("metadata", {})
                st.markdown(f"**Score**: {score:.4f}")
                st.write(metadata.get("text", "⚠️ No text found in metadata"))
        except Exception as e:
            st.error(f"❌ An error occurred while querying: {e}")
