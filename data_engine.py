import pandas as pd
import numpy as np
import os
import streamlit as st


@st.cache_data
def load_and_process_data(path):
    """Carga, limpia y genera métricas por ID."""
    if not os.path.exists(path):
        return None, None

    # 1. Carga inicial
    df = pd.read_csv(path)

    # 2. Limpieza de Educación y booleanos
    df['Education'] = df['Education'].astype(str).str.lower().str.strip()
    edu_mapping = {
        "high school": 1,
        "graduate": 2,
        "postgraduate": 3,
        "master and doctor": 4
    }
    df['Education'] = df['Education'].map(edu_mapping).fillna(1).astype(int)
    df['Social_drinker'] = df['Social_drinker'].astype(int)
    df['Social_smoker'] = df['Social_smoker'].astype(int)

    # 3. Agrupación por Empleado (ID)
    df_id = df.groupby('ID').agg({
        'Hit_target': 'mean',
        'Work_load_Average_day': 'mean',
        'Disciplinary_failure': 'max',
        'Absenteeism_hours': 'sum',
        'Service_time': 'max',
        'Age': 'mean',
        'Education': 'max',
        'Son': 'max',
        'Pet': 'max',
        'Body_mass_index': 'mean',
        'Social_drinker': 'max',
        'Social_smoker': 'max',
        'Transportation_expense': 'mean',
        'Distance_Residence_Work': 'mean'
    }).reset_index()

    # 4. Creación de KPIs estratégicos
    df_id['Resiliencia_Score'] = df_id['Hit_target'] / \
        ((df_id['Work_load_Average_day'] / 100) + 0.1)
    df_id['Indice_Lealtad'] = df_id['Service_time'] / \
        (df_id['Absenteeism_hours'] + 1)

    return df, df_id
