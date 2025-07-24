import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="MLOps House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le style
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🏠 MLOps House Price Predictor</h1>
    <p>Prédiction intelligente de prix immobilier avec Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar stylée
with st.sidebar:
    st.markdown("### 🏗️ Caractéristiques de la Propriété")
    
    # Section Structure
    st.markdown("**🏠 Structure**")
    bedrooms = st.selectbox("Chambres", options=[1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Salles de bain", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=3)
    floors = st.selectbox("Étages", options=[1.0, 1.5, 2.0, 2.5, 3.0], index=2)
    
    st.markdown("---")
    
    # Section Surface
    st.markdown("**📐 Surfaces (pieds carrés)**")
    sqft_living = st.slider("Surface habitable", 500, 5000, 1800, 50)
    sqft_lot = st.slider("Surface terrain", 1000, 20000, 5000, 100)
    sqft_above = st.slider("Surface au-dessus", 500, 4000, 1600, 50)
    sqft_basement = st.slider("Surface sous-sol", 0, 2000, 200, 50)
    
    st.markdown("---")
    
    # Section Qualité
    st.markdown("**⭐ Qualité & Vues**")
    waterfront = st.toggle("🌊 Vue sur l'eau", value=False)
    view = st.slider("Qualité de la vue (0-4)", 0, 4, 2)
    condition = st.slider("État général (1-5)", 1, 5, 3)
    
    st.markdown("---")
    
    # Section Localisation
    st.markdown("**📍 Localisation**")
    city = st.selectbox("Ville", ["Seattle", "Bellevue", "Redmond", "Kirkland", "Tacoma"])
    statezip = st.selectbox("Code postal", ["WA 98101", "WA 98102", "WA 98103", "WA 98104"])

# Corps principal
col1, col2 = st.columns([2, 1])

with col1:
    # Visualisation des caractéristiques
    st.markdown("### 📊 Résumé des Caractéristiques")
    
    # Graphique radar
    categories = ['Chambres', 'S.d.B', 'Étages', 'Vue', 'État']
    values = [bedrooms/6*100, bathrooms/5*100, floors/3*100, view/4*100, condition/5*100]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Propriété',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=400,
        title="Score des Caractéristiques (%)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparaison des surfaces
    surface_data = pd.DataFrame({
        'Type': ['Habitable', 'Au-dessus', 'Sous-sol', 'Terrain'],
        'Surface': [sqft_living, sqft_above, sqft_basement, sqft_lot],
        'Couleur': ['#667eea', '#764ba2', '#f093fb', '#f5f7fa']
    })
    
    fig2 = px.bar(surface_data, x='Type', y='Surface', 
                  title="Répartition des Surfaces (sqft)",
                  color='Type',
                  color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#a8edea'])
    fig2.update_layout(showlegend=False, height=300)
    
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Carte de prédiction
    st.markdown("### 🔮 Prédiction")
    
    # Bouton de prédiction stylé
    predict_button = st.button("🚀 PRÉDIRE LE PRIX", use_container_width=True)
    
    if predict_button:
        # Données à envoyer
        data = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "city": city,
            "statezip": statezip,
            "country": "USA"
        }
        
        with st.spinner('🤖 IA en cours de calcul...'):
            time.sleep(1)  # Animation
            
            try:
                # Appel API
                response = requests.post("http://localhost:8000/predict", json=data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    price = result['predicted_price']
                    
                    # Affichage du résultat avec style
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>💰 ${price:,.0f}</h2>
                        <p>Prix estimé</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métriques détaillées
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Prix/sqft", f"${price/sqft_living:.0f}", delta=None)
                    with col_b:
                        st.metric("Modèle", result['model_name'], delta=None)
                    
                    # Informations techniques
                    with st.expander("📋 Détails Techniques"):
                        st.write(f"**Temps de traitement:** {result['processing_time_ms']:.1f}ms")
                        st.write(f"**Timestamp:** {result['timestamp']}")
                        st.write(f"**Intervalle de confiance:**")
                        st.write(f"  - Min: ${price*0.85:,.0f}")
                        st.write(f"  - Max: ${price*1.15:,.0f}")
                    
                    # Graphique de comparaison
                    comparable_prices = [price*0.85, price, price*1.15]
                    labels = ['Fourchette Basse', 'Estimation', 'Fourchette Haute']
                    
                    fig3 = go.Figure(data=[
                        go.Bar(x=labels, y=comparable_prices,
                               marker_color=['#ff7675', '#667eea', '#00b894'])
                    ])
                    fig3.update_layout(
                        title="Fourchette de Prix",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                else:
                    st.error(f"❌ Erreur API: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connexion à l'API impossible")
                st.write("Assurez-vous que l'API tourne avec: `python main.py api`")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    # État de l'API
    st.markdown("### 🔧 État de l'API")
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API Opérationnelle")
            health_data = health_response.json()
            st.write(f"**Status:** {health_data.get('status', 'Unknown')}")
        else:
            st.warning("⚠️ API Problème")
    except:
        st.error("❌ API Hors Ligne")
        st.write("Lancez: `python main.py api`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🏠 MLOps House Price Predictor</h4>
    <p>Développé avec ❤️ | Streamlit + FastAPI + Scikit-Learn</p>
    <p><strong>Instructions:</strong> Ajustez les paramètres dans la sidebar et cliquez sur 'Prédire le Prix'</p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🚀 Actions Rapides")
    
    if st.button("🔄 Valeurs par Défaut"):
        st.experimental_rerun()
    
    st.markdown("### 📖 Guide")
    with st.expander("Comment utiliser"):
        st.write("""
        1. **Ajustez** les caractéristiques dans les sections
        2. **Cliquez** sur 'Prédire le Prix'  
        3. **Analysez** les résultats et graphiques
        4. **Expérimentez** avec différentes valeurs
        """)
    
    st.markdown("### 🎯 Exemples")
    if st.button("🏠 Maison Standard"):
        st.session_state.update({
            'bedrooms': 3, 'bathrooms': 2.0, 'sqft_living': 1800
        })
    
    if st.button("🏰 Maison Luxe"):
        st.session_state.update({
            'bedrooms': 5, 'bathrooms': 4.0, 'sqft_living': 4000
        })