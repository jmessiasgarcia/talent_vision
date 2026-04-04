import streamlit as st


def apply_visual_config():
    # 1. Configuración de la página (DEBE SER LO PRIMERO)
    st.set_page_config(page_title="RRHH Analytics Dashboard", layout="wide")

    # 2. Inyección de CSS forzado
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');

        /* Forzamos el fondo en la aplicación y en el contenedor principal */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #EEEEEE !important;
        }

        /* Quitamos decoraciones y limpiamos cabecera */
        [data-testid="stDecoration"] { display: none; }
        header[data-testid="stHeader"] { background: transparent !important; }
        .block-container { padding-top: 2rem !important; }

        /* Estilo de métricas */
        [data-testid="stMetricValue"] {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            color: #353639 !important;
        }
        
        [data-testid="stMetricLabel"] p {
            color: #353639 !important;
            font-size: 1rem !important;
        }

        /* Estilo para que los widgets (selectbox, etc) se vean bien sobre el gris */
        .stSelectbox, .stTextInput {
            color: #353639;
        }
        </style>
        """, unsafe_allow_html=True)
