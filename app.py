import os
import streamlit as st
import pandas as pd
from openai import OpenAI
import pinecone

# --- Setup secrets ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX_NAME = st.secrets["PINECONE_INDEX"]

# --- Setup OpenAI Client ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Initialize Pinecone ---
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# --- Sample dataset ---
data = {
    "id": [1, 2, 3, 4],
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
}
df = pd.DataFrame(data)

# --- Streamlit UI ---
st.title("📦 Pinecone RAG Demo – Microsoft Fabric Dataset")
st.markdown("Load sample data, embed it, and ask questions using GPT and Pinecone Vector DB.")

# --- Display Sample Data ---
st.subheader("📄 Sample data loaded:")
st.dataframe(df)

# --- Embed & Upload to Pinecone ---
if st.button("🔄 Embed & Upload to Pinecone"):
    with st.spinner("Embedding and uploading to Pinecone..."):
        vectors = []
        for i, row in df.iterrows():
            response = client.embeddings.create(
                input=[row["text"]],
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            vectors.append((f"id-{i}", embedding, row["metadata"]))
        index.upsert(vectors)
    st.success("✅ Data embedded and uploaded to Pinecone.")

# --- Question Input ---
st.markdown("🔍 Ask a question about Microsoft Fabric:")
question = st.text_input("Explain the components of Microsoft Fabric with examples.")

# --- Answer Using GPT + Pinecone ---
if question:
    with st.spinner("Fetching answer..."):
        # Embed the question
        query_response = client.embeddings.create(
            input=[question],
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding

        # Query Pinecone
        result = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        # Prepare context from results
        contexts = [match['metadata']['source'] + ": " + df.iloc[int(match['id'].split('-')[1])]['text'] for match in result.matches]
        context_text = "\n".join(contexts)

        # Compose prompt
        prompt = f"Answer the following based on the context below:\n\nContext:\n{context_text}\n\nQuestion: {question}"

        # Generate response
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who answers questions based on context."},
                {"role": "user", "content": prompt}
            ]
        )
        st.markdown("### 🧠 GPT Answer:")
        st.write(completion.choices[0].message.content)
