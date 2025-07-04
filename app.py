import os
import streamlit as st
import pandas as pd
import pinecone
from uuid import uuid4

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

# Connect to Pinecone (modern v2 client)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)

# Streamlit UI
st.title("📄 Sample Data Loader + Pinecone Uploader")
st.markdown("Load sample data, embed to Pinecone, and test questions using GPT + vector search.")

# Sample data
df = pd.DataFrame({
    "id": [str(i) for i in range(1, 5)],
    "text": [
        "Microsoft Fabric is an end-to-end, SaaS analytics platform.",
        "Power BI is a part of Microsoft Fabric used for visual analytics.",
        "Data Factory in Fabric helps in orchestrating data workflows.",
        "OneLake is the unified data lake for all Fabric workloads."
    ],
    "metadata": [
        {"source": "fabric_intro"},
        {"source": "power_bi"},
        {"source": "data_factory"},
        {"source": "onelake"}
    ]
})

st.subheader("📊 Sample data loaded:")
st.dataframe(df)

# Upload to Pinecone (no embedding required)
if st.button("🔗 Embed & Upload to Pinecone"):
    with st.spinner("Uploading to Pinecone..."):
        try:
            # Construct documents for auto-embedding (only id, text, metadata)
            to_upsert = [
                {
                    "id": row["id"],
                    "values": None,  # Skip manual vector, Pinecone auto-generates from 'text'
                    "metadata": {
                        "text": row["text"],
                        **row["metadata"]
                    }
                }
                for _, row in df.iterrows()
            ]
            index.upsert(vectors=to_upsert)
            st.success("✅ Uploaded to Pinecone successfully!")
        except Exception as e:
            st.error(f"❌ Failed to upload: {e}")
