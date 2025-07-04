import streamlit as st
import pandas as pd
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

# Secrets from Streamlit or .env
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = "us-east-1"
PINECONE_INDEX = "fabric-index"

# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX)

# Sample data
data = [
    {"id": "1", "text": "Microsoft Fabric is an end-to-end, SaaS analytics platform.", "metadata": {"source": "fabric_intro"}},
    {"id": "2", "text": "Power BI is a part of Microsoft Fabric used for visual analytics.", "metadata": {"source": "power_bi"}},
    {"id": "3", "text": "Data Factory in Fabric helps in orchestrating data workflows.", "metadata": {"source": "data_factory"}},
    {"id": "4", "text": "OneLake is the unified data lake for all Fabric workloads.", "metadata": {"source": "onelake"}},
]
df = pd.DataFrame(data)

# Streamlit UI
st.title("📄 Upload & Embed to Pinecone")
st.subheader("✅ Sample data loaded:")
st.dataframe(df)

if st.button("📥 Embed & Upload to Pinecone"):
    st.info("Embedding text and uploading to Pinecone...")

    vectors = []
    for row in data:
        # Generate embedding
        response = openai.embeddings.create(
            input=row["text"],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        # Build Pinecone vector
        vectors.append({
            "id": row["id"],
            "values": embedding,
            "metadata": row["metadata"]
        })

    # Upload to Pinecone
    index.upsert(vectors=vectors)
    st.success("✅ Uploaded successfully to Pinecone!")
