import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import fitz
import ollama
import chromadb
import joblib
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
PDF_PATHS = [
    "data/IOC_LEAFLET_ZERO-ADD_WEB-3.pdf",
    "data/guide-liste-ingredients-nutritionnelle-des-vins.pdf",
]
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3.5"
COLLECTION_NAME = "wine_knowledge"
MODEL_PATH = "wine_quality_model.pkl"
SCALER_PATH = "wine_quality_scaler.pkl"

SYSTEM_PROMPT = """Tu es un expert en œnologie et en chimie du vin. Tu réponds aux questions en te basant UNIQUEMENT sur le contexte fourni.
Règles :
- Réponds de manière détaillée et structurée en français.
- Cite les pages sources entre parenthèses (ex: (p.12)).
- Si l'information n'est pas dans le contexte, dis-le explicitement.
- Utilise des listes à puces pour structurer ta réponse quand c'est pertinent.
- Si une prédiction ML est fournie, enrichis-la avec des explications chimiques tirées du contexte."""


# --- RAG Core ---
@st.cache_resource
def load_collection():
    chunks = []
    for pdf_path in PDF_PATHS:
        doc = fitz.open(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                for chunk in text_splitter.split_text(text):
                    chunks.append({
                        "text": chunk,
                        "page": i + 1,
                        "source": os.path.basename(pdf_path),
                    })
        doc.close()

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = [get_embedding(c["text"]) for c in batch]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"], "source": c["source"]} for c in batch],
        )

    return collection


def get_embedding(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def retrieve(collection, question: str, n_results: int = 6) -> list[dict]:
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "page": meta["page"],
            "source": meta["source"],
            "distance": dist,
        })
    return retrieved


def rag(collection, question: str, prediction_context: str = "", n_results: int = 6) -> str:
    retrieved_chunks = retrieve(collection, question, n_results=n_results)

    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[{chunk['source']} — p.{chunk['page']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    pred_section = f"\n\nPrédiction du modèle ML : {prediction_context}" if prediction_context else ""

    user_prompt = f"""Contexte extrait de la base de connaissances sur le vin :
{context}{pred_section}

---

Question : {question}

Réponse détaillée :"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


# --- ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception:
        return None, None


def predict_quality(model, scaler, features: dict) -> int | None:
    if model is None:
        return None
    cols_to_scale = [c for c in features if c != "type"]
    df = pd.DataFrame([features])
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return int(model.predict(df_scaled)[0])


# --- Streamlit UI ---
st.set_page_config(page_title="Wine Expert Chatbot", page_icon="🍷", layout="wide")
st.title("🍷 Wine Expert Chatbot — RAG + ML")
st.write(
    "Posez une question sur le vin. Vous pouvez aussi renseigner les caractéristiques "
    "chimiques d'un vin pour obtenir une prédiction ML enrichie par les connaissances expertes."
)

with st.spinner("Chargement de la base de connaissances..."):
    collection = load_collection()
    ml_model, ml_scaler = load_ml_model()

# --- Sidebar : ML prediction ---
with st.sidebar:
    st.header("🔬 Prédiction ML (optionnel)")
    st.caption("Renseignez les caractéristiques du vin pour enrichir la réponse.")

    wine_type = st.selectbox("Type", ["Rouge (1)", "Blanc (0)"])
    fixed_acidity      = st.number_input("Fixed acidity",      value=7.0,  step=0.1)
    volatile_acidity   = st.number_input("Volatile acidity",   value=0.5,  step=0.01)
    citric_acid        = st.number_input("Citric acid",        value=0.3,  step=0.01)
    residual_sugar     = st.number_input("Residual sugar",     value=2.0,  step=0.1)
    chlorides          = st.number_input("Chlorides",          value=0.08, step=0.001, format="%.3f")
    free_sulfur        = st.number_input("Free sulfur dioxide",value=30.0, step=1.0)
    total_sulfur       = st.number_input("Total sulfur dioxide",value=100.0,step=1.0)
    density            = st.number_input("Density",            value=0.996,step=0.001, format="%.3f")
    ph                 = st.number_input("pH",                 value=3.3,  step=0.01)
    sulphates          = st.number_input("Sulphates",          value=0.6,  step=0.01)
    alcohol            = st.number_input("Alcohol",            value=10.0, step=0.1)

    features = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur,
        "total sulfur dioxide": total_sulfur,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "type": 1 if "Rouge" in wine_type else 0,
    }

    pred_quality = None
    if ml_model is not None:
        pred_quality = predict_quality(ml_model, ml_scaler, features)
        st.success(f"Qualité prédite : **{pred_quality} / 10**")
    else:
        st.warning("Modèle ML non trouvé. Lancez d'abord les cellules 4.2 et 4.6.")

# --- Main chat ---
question = st.text_input("Votre question :", placeholder="Ex: Quel est l'impact de l'acidité volatile sur la qualité du vin ?")

if question:
    prediction_context = ""
    if pred_quality is not None:
        prediction_context = (
            f"Le modèle ML prédit une qualité de {pred_quality}/10 pour un vin "
            f"{'rouge' if features['type'] == 1 else 'blanc'} avec les caractéristiques suivantes : "
            + ", ".join(f"{k}={v}" for k, v in features.items() if k != "type")
            + "."
        )

    with st.spinner("Recherche et génération de la réponse..."):
        answer = rag(collection, question, prediction_context=prediction_context)

    st.markdown("### Réponse")
    st.markdown(answer)

    with st.expander("📄 Sources utilisées"):
        chunks = retrieve(collection, question)
        for c in chunks:
            st.markdown(f"**{c['source']} — p.{c['page']}** *(distance: {c['distance']:.3f})*")
            st.caption(c["text"][:300] + "...")
