import streamlit as st
import joblib
import numpy as np
import pandas as pd
 
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="centered",
)
 
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model():
    model  = joblib.load("wine_quality_model.pkl")
    scaler = joblib.load("wine_quality_scaler.pkl")
    return model, scaler
 
model, scaler = load_model()
 
st.title("Wine Quality Predictor")
st.markdown("Renseigne les composantes chimiques du vin pour prédire sa qualité (note de **0 à 10**).")
st.divider()
col1, col2 = st.columns(2)
 
with col1:
    fixed_acidity      = st.slider("Fixed Acidity",        3.8,  16.0,  7.4,  step=0.1)
    volatile_acidity   = st.slider("Volatile Acidity",     0.08,  1.6,  0.52, step=0.01)
    citric_acid        = st.slider("Citric Acid",          0.0,   1.7,  0.26, step=0.01)
    residual_sugar     = st.slider("Residual Sugar",       0.6,  66.0,  2.2,  step=0.1)
    chlorides          = st.slider("Chlorides",            0.009, 0.62, 0.047, step=0.001, format="%.3f")
    free_so2           = st.slider("Free Sulfur Dioxide",  1.0, 290.0, 35.0,  step=1.0)
 
with col2:
    total_so2          = st.slider("Total Sulfur dioxide", 6.0, 440.0, 120.0, step=1.0)
    density            = st.slider("Density",              0.987, 1.039, 0.9946, step=0.0001, format="%.4f")
    ph                 = st.slider("pH",                   2.72,  4.01,  3.21,  step=0.01)
    sulphates          = st.slider("Sulphates",            0.22,  2.0,   0.53,  step=0.01)
    alcohol            = st.slider("Alcohol (%)",          8.0,  15.0,  10.4,  step=0.1)
    wine_type          = st.selectbox("Type de vin", ["Rouge", "Blanc"])
 
st.divider()
 
if st.button("🔍 Prédire la qualité", use_container_width=True, type="primary"):
    type_encoded = 1 if wine_type == "Rouge" else 0

    cols_to_scale = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    sample = pd.DataFrame([{
        'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity,
        'citric acid': citric_acid, 'residual sugar': residual_sugar,
        'chlorides': chlorides, 'free sulfur dioxide': free_so2,
        'total sulfur dioxide': total_so2, 'density': density,
        'pH': ph, 'sulphates': sulphates, 'alcohol': alcohol, 'type': type_encoded
    }])
    sample[cols_to_scale] = scaler.transform(sample[cols_to_scale])
    prediction = model.predict(sample)[0]
 
    if prediction <= 4:
        color, label = "#e74c3c", "Mauvaise qualité"
    elif prediction <= 6:
        color, label = "#f39c12", "Qualité moyenne"
    else:
        color, label = "#2ecc71", "Bonne qualité"
 
    st.markdown(
        f"""
        <div style="
            background-color:{color}22;
            border-left: 5px solid {color};
            padding: 20px 24px;
            border-radius: 8px;
            margin-top: 16px;
        ">
            <h2 style="color:{color}; margin:0;">Note prédite : {prediction} / 10</h2>
            <p style="color:{color}; margin:4px 0 0 0; font-size:1.1em;">{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    with st.expander("Voir les valeurs saisies"):
        recap = pd.DataFrame({
            "Variable": [
                "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
                "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol", "Type"
            ],
            "Valeur": [
                str(fixed_acidity), str(volatile_acidity), str(citric_acid), str(residual_sugar),
                str(chlorides), str(free_so2), str(total_so2), str(density), str(ph),
                str(sulphates), str(alcohol), wine_type
            ]
        })
        st.dataframe(recap, use_container_width=True, hide_index=True)