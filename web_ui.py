import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Variables globales
MODEL_PATH = "models/model.pkl"
API_URL = "http://localhost:8000"

@st.cache_resource
def load_direct_model():
    """Charge le modèle directement depuis le fichier"""
    try:
        if os.path.exists(MODEL_PATH):
            model_package = joblib.load(MODEL_PATH)
            return model_package
        else:
            st.error(f"Modèle non trouvé: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

def check_api_available():
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def predict_with_direct_model(model_package, house_data):
    """Prédiction directe avec le modèle local"""
    try:
        # Créer DataFrame
        df = pd.DataFrame([house_data])
        
        # Feature engineering (même logique que dans api.py)
        df['house_age'] = 2024 - 1990
        df['is_renovated'] = 0
        df['total_sqft'] = df['sqft_living'] + df['sqft_basement']
        df['waterfront'] = df['waterfront'].astype(int)
        
        # Encoder variables catégorielles si disponible
        if 'encoders' in model_package:
            encoders = model_package['encoders']
            for col, encoder in encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform([df[col].iloc[0]])[0]
                    except:
                        df[col] = 0
        
        # Sélectionner features dans le bon ordre
        feature_names = model_package['feature_names']
        X = df[feature_names].values
        
        # Normaliser
        X_scaled = model_package['scaler'].transform(X)
        
        # Prédiction
        prediction = model_package['model'].predict(X_scaled)[0]
        prediction = max(50000, float(prediction))
        
        return {
            'predicted_price': prediction,
            'model_name': model_package['model_name'],
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def predict_with_api(house_data):
    """Prédiction via API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=house_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f"API Error: {response.status_code}", 'success': False}
    except Exception as e:
        return {'error': str(e), 'success': False}

def main():
    """Interface principale"""
    
    st.title("🏠 Prédiction de Prix Immobilier")
    st.markdown("---")
    
    # Sidebar pour les informations
    with st.sidebar:
        st.header("📊 Informations")
        
        # Statut du modèle
        model_package = load_direct_model()
        if model_package:
            st.success("✅ Modèle local chargé")
            st.info(f"**Modèle:** {model_package.get('model_name', 'Unknown')}")
            st.info(f"**R² Score:** {model_package.get('performance', {}).get('r2_test', 'N/A'):.3f}")
        else:
            st.error("❌ Modèle local non disponible")
        
        # Statut de l'API
        api_available = check_api_available()
        if api_available:
            st.success("✅ API disponible")
        else:
            st.warning("⚠️ API non disponible")
        
        # Mode de prédiction
        st.header("🔧 Mode de Prédiction")
        if model_package and api_available:
            prediction_mode = st.radio(
                "Choisir le mode:",
                ["Modèle Direct", "Via API", "Comparaison"]
            )
        elif model_package:
            prediction_mode = "Modèle Direct"
            st.info("Mode: Modèle Direct uniquement")
        elif api_available:
            prediction_mode = "Via API" 
            st.info("Mode: API uniquement")
        else:
            st.error("Aucun mode disponible")
            return
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🏡 Caractéristiques de la Maison")
        
        # Formulaire
        with st.form("house_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                bedrooms = st.number_input("Chambres", min_value=1, max_value=10, value=3)
                bathrooms = st.number_input("Salles de bain", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
                sqft_living = st.number_input("Surface habitable (sqft)", min_value=500, max_value=10000, value=1800)
                sqft_lot = st.number_input("Surface terrain (sqft)", min_value=1000, max_value=50000, value=5000)
                floors = st.number_input("Étages", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
            
            with col_b:
                waterfront = st.checkbox("Vue sur l'eau")
                view = st.slider("Qualité de la vue", 0, 4, 2)
                condition = st.slider("État de la maison", 1, 5, 3)
                sqft_above = st.number_input("Surface au-dessus sol (sqft)", min_value=500, max_value=8000, value=1600)
                sqft_basement = st.number_input("Surface sous-sol (sqft)", min_value=0, max_value=3000, value=200)
            
            city = st.text_input("Ville", value="Seattle")
            statezip = st.text_input("État/Code postal", value="WA 98101")
            
            submitted = st.form_submit_button("🔮 Prédire le Prix", use_container_width=True)
    
    with col2:
        st.header("💰 Résultat")
        
        if submitted:
            # Préparer les données
            house_data = {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft_living': sqft_living,
                'sqft_lot': sqft_lot,
                'floors': floors,
                'waterfront': waterfront,
                'view': view,
                'condition': condition,
                'sqft_above': sqft_above,
                'sqft_basement': sqft_basement,
                'city': city,
                'statezip': statezip,
                'country': 'USA'
            }
            
            # Prédictions selon le mode
            if prediction_mode == "Modèle Direct":
                with st.spinner("Prédiction en cours..."):
                    result = predict_with_direct_model(model_package, house_data)
                    
                if result['success']:
                    st.success(f"**${result['predicted_price']:,.0f}**")
                    st.info(f"Modèle: {result['model_name']}")
                else:
                    st.error(f"Erreur: {result['error']}")
            
            elif prediction_mode == "Via API":
                with st.spinner("Prédiction via API..."):
                    result = predict_with_api(house_data)
                    
                if 'predicted_price' in result:
                    st.success(f"**${result['predicted_price']:,.0f}**")
                    st.info(f"Modèle: {result['model_name']}")
                    st.caption(f"Temps: {result['processing_time_ms']:.1f}ms")
                else:
                    st.error(f"Erreur API: {result.get('error', 'Unknown')}")
            
            elif prediction_mode == "Comparaison":
                col_direct, col_api = st.columns(2)
                
                with col_direct:
                    st.subheader("Modèle Direct")
                    with st.spinner("Prédiction..."):
                        result_direct = predict_with_direct_model(model_package, house_data)
                    
                    if result_direct['success']:
                        st.success(f"${result_direct['predicted_price']:,.0f}")
                    else:
                        st.error("Erreur")
                
                with col_api:
                    st.subheader("Via API")
                    with st.spinner("Prédiction..."):
                        result_api = predict_with_api(house_data)
                    
                    if 'predicted_price' in result_api:
                        st.success(f"${result_api['predicted_price']:,.0f}")
                        if result_direct['success']:
                            diff = abs(result_api['predicted_price'] - result_direct['predicted_price'])
                            st.caption(f"Différence: ${diff:,.0f}")
                    else:
                        st.error("Erreur API")
    
    # Graphiques de démonstration
    st.markdown("---")
    st.header("📈 Analyse des Prix")
    
    if st.button("Générer Analyse Démo"):
        # Créer des données de démonstration
        demo_data = []
        for i in range(50):
            demo_data.append({
                'sqft_living': np.random.randint(1000, 4000),
                'price': np.random.randint(200000, 800000),
                'bedrooms': np.random.randint(2, 5)
            })
        
        df_demo = pd.DataFrame(demo_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(df_demo, x='sqft_living', y='price', 
                            title="Prix vs Surface habitable",
                            labels={'sqft_living': 'Surface (sqft)', 'price': 'Prix ($)'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.box(df_demo, x='bedrooms', y='price',
                         title="Distribution des prix par chambres")
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()