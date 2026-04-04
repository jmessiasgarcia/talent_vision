# 1. Librerías Base
from kpi_engine import calculate_general_kpis
import os
import pandas as pd
import numpy as np
from data_engine import load_and_process_data
from stats_engine import run_statistical_analysis, run_predictive_model
from visuals import apply_visual_config
from ia_engine import run_ia_pipeline

# 2. Interfaz y Visualización
import streamlit as st
import plotly.express as px
from visuals import apply_visual_config

# 3. Ciencia de Datos y ML
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


# 4. Estadística
import scipy.stats as stats
from scipy.stats import (
    pearsonr,
    spearmanr,
    ttest_ind,
    pointbiserialr,
    f_oneway,
    chi2_contingency,
)

apply_visual_config()

base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'full_RRHH.csv')
RRHH_full, df_id = load_and_process_data(csv_path)
if RRHH_full is None:
    st.error("❌ No se encontró el archivo CSV")
    st.stop()

total_empleados_unicos = df_id['ID'].nunique()

df_id, score_sil, score_db, X_scaled = run_ia_pipeline(df_id)
df_hit_stat, df_disc_stat, df_results = run_statistical_analysis(df_id)
df_importance = run_predictive_model(RRHH_full)
df_disc = df_disc_stat.sort_values('Impacto', ascending=True)
df_hit = df_hit_stat.sort_values('Impacto', ascending=True)


st.title("Rapid Express: People Analytics Dashboard")
st.markdown(
    f"### Porque detrás de cada ruta, hay una historia: cuidamos de las **{total_empleados_unicos}** personas que mueven nuestro motor.")

counts = df_id['Segmento'].value_counts()
total_a = counts.get("Motor Familiar", 0)
total_b = counts.get("Talento Enfocado", 0)
total_c = counts.get("Talento Senior", 0)

st.write("### Distribución de la Fuerza Laboral")
col1, col2, col3 = st.columns(3)

col1.metric("Motor Familiar", f"{total_a} talentos", delta="42%")
col2.metric("Talento Enfocado", f"{total_b} talentos", delta="39%")
col3.metric("Talento Senior", f"{total_c} talentos", delta="19%")

st.divider()


st.write("### Calidad del Modelo Predictivo")
m1, m2 = st.columns(2)
m1.metric("Silhouette Score", f"{score_sil:.2f}")
m2.metric("Davies-Bouldin Index", f"{score_db:.2f}")

orden_segmentos = [
    "Motor Familiar",
    "Talento Enfocado",
    "Talento Senior"
]

opciones = ["Todos los empleados"] + orden_segmentos

segmento_seleccionado = st.selectbox(
    "Filtrar por segmento",
    opciones
)

# ==========================================
# DATAFRAME PARA VISUALIZACIÓN
# ==========================================

df_plot = df_id.copy()

df_plot["PCA1_jitter"] = df_plot["PCA1"] + \
    np.random.uniform(-0.1, 0.1, len(df_plot))
df_plot["PCA2_jitter"] = df_plot["PCA2"] + \
    np.random.uniform(-0.1, 0.1, len(df_plot))

if segmento_seleccionado != "Todos los empleados":
    df_plot = df_plot[df_plot["Segmento"] == segmento_seleccionado]


# ==========================================
# MAPA DE SÍMBOLOS
# ==========================================

simbolos_map = {
    "Motor Familiar": "circle",
    "Talento Enfocado": "diamond",
    "Talento Senior": "square"
}

# ==========================================
# SCATTER PCA
# ==========================================

fig_scatter = px.scatter(
    df_plot,
    x="PCA1_jitter",
    y="PCA2_jitter",
    color="Segmento",
    symbol="Segmento",
    symbol_map=simbolos_map,
    size=[12] * len(df_plot),
    text="ID",
    title="Entendiendo a nuestra gente para cuidarlos mejor",
    hover_name="ID",
    hover_data={
        "PCA1_jitter": False,
        "PCA2_jitter": False,
        "Age": True,
        "Son": True,
        "Social_smoker": True,
        "Social_drinker": True,
        "Education": True,
        "Body_mass_index": ':.1f'
    },
    color_discrete_sequence=["#51A242", "#65E74B", "#093C2B"],
    template="plotly_white",
    height=600
)

# ==========================================
# ESTILO DEL GRÁFICO
# ==========================================

fig_scatter.update_layout(
    plot_bgcolor="#EEEEEE",
    paper_bgcolor="#EEEEEE",
    xaxis=dict(
        title=None,
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False
    ),
    yaxis=dict(
        title=None,
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False
    ),

    margin=dict(l=20, r=20, t=40, b=20)
)


fig_scatter.update_traces(
    mode="markers",
    marker=dict(
        size=22,
        line=dict(width=0.5, color="black"),
        opacity=0.5
    )
)


st.plotly_chart(fig_scatter, width='stretch')

with st.expander("¿Cómo leer este mapa de segmentación?", expanded=False):
    st.markdown("""
    Este gráfico utiliza **Inteligencia Artificial (K-Means + PCA)** para agrupar a los empleados según sus perfiles sociodemográficos únicos.

    ###  Variables analizadas:
    Para crear estos grupos, el modelo ha procesado 6 dimensiones clave de cada empleado:
    1. **Edad** (`Age`): Media de edad del trabajador.
    2. **Carga Familiar** (`Son`): Número de hijos.
    3. **Salud Física** (`Body_mass_index`): Índice de masa corporal promedio.
    4. **Educación** (`Education`): Nivel de estudios alcanzado.
    5. **Hábitos Sociales** (`Social_drinker` y `Social_smoker`): Si el empleado fuma o consume alcohol habitualmente.

    ###  Guía de interpretación:
    * **Ejes (PCA1 y PCA2):** Son "super-variables" que combinan las 6 anteriores. Si dos puntos están **cerca**, significa que esos empleados tienen vidas y hábitos muy parecidos.
    * **Colores (Segmentos):** Representan los 3 perfiles estratégicos identificados (Motor Familiar, Talento Enfocado y Senior).
    * **Interactividad:** Puedes hacer zoom, aislar segmentos haciendo clic en la leyenda o pasar el ratón para ver el **ID del empleado**.
    """)


with st.expander("Ver detalle de decisiones estratégicas por Segmento"):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### El Motor Familiar")
        st.info("**42%**")
        st.write("""
        * **Perfil:** Media de 37 años, alta carga familiar (1.4 hijos) y estilo de vida sedentario por falta de tiempo.
        * **Riesgos:** Burnout por fricción vida-trabajo, estrés logístico y fatiga física acumulada.
        * **Palanca:** Conciliación Efectiva.
        * **Acciones:** Smart Working y flexibilidad horaria escolar.
            * Beneficios como cheques guardería.
            * Rutas recurrentes (menor carga cognitiva) y formación en ergonomía.
        """)

    with col_b:
        st.markdown("### Talento Enfocado")
        st.info("**39%**")
        st.write("""
        * **Perfil:** Saludable (bajo IMC), sin cargas familiares, alta energía y disponibilidad.
        * **Riesgos:** Fuga de talento por falta de retos y estancamiento en tareas repetitivas.
        * **Palanca:** Desarrollo y Upskilling.
        * **Acciones:** Planes de carrera vertical/horizontal y certificación "Senior".
            * Ranking mensual de eficiencia con incentivos por volumen.
            * Promoción a coordinadores de zona o proyectos piloto.
        """)

    with col_c:
        st.markdown("### Talento Senior")
        st.info("**19%**")
        st.write("""
        * **Perfil:** Mayor edad, fumadores (gestión de estrés de riesgo) y amplia experiencia.
        * **Riesgos:** Fatiga física, lesiones y accidentes por desgaste acumulado.
        * **Palanca:** Bienestar Integral y Reconocimiento.
        * **Acciones:** * Programa de fisioterapia preventiva y deshabituación tabáquica.
            * Transición a roles de **Mentoring**.
            * Adaptación de rutas (menos entregas) y vehículos automáticos.
        """)


with st.expander("Datos del trabajador", expanded=False):
    df_completo_ia = df_id[[
        'ID', 'Segmento', 'Age', 'Body_mass_index', 'Son',
        'Work_load_Average_day', 'Hit_target', 'Absenteeism_hours'
    ]]

    st.dataframe(
        df_completo_ia,
        width='stretch',
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID"),
            "Segmento": st.column_config.TextColumn("Segmento"),
            "Absenteeism_time_in_hours": st.column_config.NumberColumn(
                "Total Horas Ausencia",
                help="Suma acumulada de todas las faltas",
                format="%d h 🕒",
            ),
            "Hit_target": st.column_config.ProgressColumn(
                "Cumplimiento",
                format="%d%%",
                min_value=0,
                max_value=100
            ),
            "Work_load_Average_day": st.column_config.NumberColumn(
                "Carga Media",
                format="%.0f"
            )
        }
    )

with st.expander("Ver infogramas de perfil"):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image("assets/perfil_a.jpg")
        st.caption("Motor Familiar")
    with col2:
        st.image("assets/perfil_b.jpg")
        st.caption("Talento Enfocado")
    with col3:
        st.image("assets/perfil_c.jpg")
        st.caption("Talento Senior")

distortions = []
K_range = range(1, 11)


gasto_total_transporte = RRHH_full[RRHH_full['Absenteeism_hours']
                                   > 0]['Transportation_expense'].sum()
bmi_promedio = RRHH_full['Body_mass_index'].mean()
tasa_fallos = (RRHH_full['Disciplinary_failure'].sum() / len(RRHH_full)) * 100
promedio_hijos = RRHH_full['Son'].mean()
st.write("### Análisis de Perfil y Riesgo")

nk1, nk2, nk3, nk4 = st.columns(4)

with nk1:
    st.metric(label="Gasto Transporte",
              value=f"R${gasto_total_transporte:,.0f}")
    st.caption("Asociado a días de ausencia")

with nk2:
    st.metric(label="IMC Promedio", value=f"{bmi_promedio:.1f}")
    status_bmi = "Sobrepeso" if bmi_promedio > 25 else "Normal"
    st.caption(f"Perfil: {status_bmi}")

with nk3:
    st.metric(label="Fallos Disc.",
              value=f"{tasa_fallos:.1f}%")
    st.caption("Impacto en disciplina")

with nk4:
    st.metric(label="Carga Familiar", value=f"{promedio_hijos:.1f}")
    st.caption("Promedio de hijos")

st.divider()


kpis = calculate_general_kpis(RRHH_full)

st.metric("Tasa Ausentismo", f"{kpis['tasa_abs']:.2f}%")

df_estacional = RRHH_full.merge(df_id[['ID', 'Segmento']], on='ID')
df_estacional = df_estacional[df_estacional['Month_absence'] != 0]
df_mensual = df_estacional.groupby(['Month_absence', 'Segmento'])[
    'Absenteeism_hours'].sum().reset_index()

fig_estacional = px.line(
    df_mensual,
    x='Month_absence',
    y='Absenteeism_hours',
    color='Segmento',
    title="Patrón de Ausencias en un Ciclo Anual (Ene-Dic)",
    markers=True,
    labels={'Month_absence': 'Mes del Año',
            'Absenteeism_hours': 'Total Horas de Ausencia'},
    color_discrete_sequence=["#093C2B", '#65E74B', "#51A242"]
)

fig_estacional.update_layout(
    plot_bgcolor='#EEEEEE',
    paper_bgcolor='#EEEEEE',
    font=dict(family="Inter", color="black"),
    xaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        dtick=1,
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        tickfont=dict(color='black')
    ),
    legend=dict(bgcolor='rgba(0,0,0,0)')
)

fig_estacional.update_traces(
    line=dict(width=3),
    marker=dict(size=8, line=dict(width=0, color='white'))
)

st.plotly_chart(fig_estacional, width='stretch')

st.divider()

st.divider()
st.header("Lógica & Exito: ¿Qué impulsa el rendimiento y la buena conducta?")

# Reconectamos las columnas con los datos que vienen del motor (df_disc_stat y df_hit_stat)
col1, col2 = st.columns(2)

with col1:
    # Usamos df_disc_stat (el resultado del motor para disciplina)
    fig_disc = px.bar(df_disc_stat.sort_values('Impacto', ascending=True),
                      x='Impacto', y='Variable', orientation='h',
                      title="Impacto en Disciplina",
                      color='Impacto', color_continuous_scale='RdYlGn',
                      template='plotly_white')
    fig_disc.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20),
                           coloraxis_showscale=False,
                           yaxis=dict(tickfont=dict(size=14)),
                           plot_bgcolor='#EEEEEE', paper_bgcolor='#EEEEEE')
    st.plotly_chart(fig_disc, use_container_width=True)

with col2:
    # Usamos df_hit_stat (el resultado del motor para rendimiento)
    fig_hit = px.bar(df_hit_stat.sort_values('Impacto', ascending=True),
                     x='Impacto', y='Variable', orientation='h',
                     title="Impacto en Rendimiento",
                     color='Impacto', color_continuous_scale='RdYlGn',
                     template='plotly_white')
    fig_hit.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20),
                          coloraxis_showscale=False,
                          yaxis=dict(tickfont=dict(size=14)),
                          plot_bgcolor='#EEEEEE', paper_bgcolor='#EEEEEE')
    st.plotly_chart(fig_hit, use_container_width=True)

with st.expander("Tabla de Tests y Resultados", expanded=False):
    # Usamos df_results que también viene del motor
    st.dataframe(df_results, width='stretch')


# ==========================================
# IDENTIFICACIÓN DEL TALENTO IDEAL (PERFIL TOP)
# ==========================================

df_talento_ideal = RRHH_full[
    (RRHH_full['Disciplinary_failure'] == 0) &
    (RRHH_full['Hit_target'] >= 95) &
    (RRHH_full['Absenteeism_hours'] == 0)
]

total_ideales = len(df_talento_ideal)
with st.expander("Ver Cuadro de Honor", expanded=False):
    # Filtramos: Sin fallos, Rendimiento >= 95% Y Cero Absentismo
    df_talento_ideal = RRHH_full[
        (RRHH_full['Disciplinary_failure'] == 0) &
        (RRHH_full['Hit_target'] >= 95) &
        (RRHH_full['Absenteeism_hours'] == 0)
    ]

    total_ideales = len(df_talento_ideal)

    st.write(f"### El Talento Ideal ({total_ideales} empleados)")

    if total_ideales > 0:
        st.info(
            f"Hemos identificado a {total_ideales} profesionales con conducta impecable, "
            f"rendimiento superior al 95% y 0 ausencias."
        )

        st.dataframe(
            df_talento_ideal[['ID', 'Education',
                              'Hit_target', 'Service_time', 'Age']],
            width='stretch',
            hide_index=True,
            column_config={
                "Hit_target": st.column_config.ProgressColumn("Rendimiento", format="%d%%", min_value=0, max_value=100),
                "Service_time": "Antigüedad",
                "Age": "Edad"
            }
        )
    else:
        st.warning(
            "No hay empleados que cumplan los tres criterios simultáneamente (Disciplina + Rendimiento + Presencia)."
        )

with st.expander("Explorar Dataset Completo (RRHH_full)"):
    st.write("Auditoría de datos originales.")
    search_id = st.text_input("🔍 Buscar por ID de empleado:", "")
    df_display = RRHH_full[RRHH_full['ID'].astype(
        str).str.contains(search_id)] if search_id else RRHH_full

    st.dataframe(
        df_display, width='stretch',
        column_config={
            "Hit_target": st.column_config.ProgressColumn("Rendimiento (%)", format="%f", min_value=0, max_value=100),
            "Disciplinary_failure": st.column_config.CheckboxColumn("Fallo Disciplinario")
        }
    )

    csv = RRHH_full.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv,
                       file_name='RRHH_full.csv', mime='text/csv')

st.divider()

mes_pico_idx = RRHH_full.groupby('Month_absence')[
    'Absenteeism_hours'].sum().idxmax()
meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
             7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
mes_pico_nombre = meses_map.get(mes_pico_idx, f"Mes {mes_pico_idx}")

motivo_modal = RRHH_full['Reason_Description'].mode()[0]

eficiencia_relativa = (RRHH_full['Hit_target'].mean(
) / RRHH_full['Work_load_Average_day'].mean()) * 100


total_empleados = RRHH_full['ID'].nunique()
empleados_ausentes = RRHH_full[RRHH_full['Absenteeism_hours'] > 0]['ID'].nunique(
)
porcentaje_ausencia = (empleados_ausentes / total_empleados) * 100


def clasificar_perfil_id(row):

    if row['Disciplinary_failure'] == 1 and row['Riesgo_Carga'] == 'Saturación':
        return 'Perfil de Inestabilidad'

    if row['Riesgo_Carga'] == 'Saturación' and row['Riesgo_Distancia'] != 'Local':
        return 'Saturado Logístico'

    if row['Service_time'] < 5 and row['Riesgo_Carga'] in ['Óptima', 'Saturación']:
        return 'En desarrollo operativo'

    return 'Perfil Estable'


@st.cache_data
def procesar_datos():
    RRHH = pd.read_csv("full_RRHH.csv")
    df_analisis = RRHH.copy()
    df_analisis['Riesgo_Distancia'] = pd.cut(
        df_analisis['Distance_Residence_Work'],
        bins=[0, 16, 49, 53],
        labels=['Local', 'Media', 'Crítica']
    )

    df_analisis['Riesgo_Carga'] = pd.cut(
        df_analisis['Work_load_Average_day'],
        bins=[0, 244, 284, 400],
        labels=['Ligera', 'Óptima', 'Saturación']
    )
    df_empleados = df_analisis.groupby('ID').agg({
        'Service_time': 'max',
        'Distance_Residence_Work': 'mean',
        'Work_load_Average_day': 'mean',
        'Hit_target': 'mean',
        'Disciplinary_failure': 'max',
    }).reset_index()
    df_empleados['Riesgo_Distancia'] = pd.cut(
        df_empleados['Distance_Residence_Work'],
        bins=[0, 16, 49, 53],
        labels=['Local', 'Media', 'Crítica']
    )

    df_empleados['Riesgo_Carga'] = pd.cut(
        df_empleados['Work_load_Average_day'],
        bins=[0, 244, 284, 400],
        labels=['Ligera', 'Óptima', 'Saturación']
    )

    df_empleados['Perfil_Riesgo'] = df_empleados.apply(
        clasificar_perfil_id, axis=1
    )

    return df_analisis, df_empleados


try:
    df_analisis, df_empleados = procesar_datos()
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo 'full_RRHH.csv'. Por favor, cárgalo en la carpeta del proyecto.")
    st.stop()

data = {
    'Semana': ['Semana 1', 'Semana 2', 'Semana 3'],
    'Ausencia (%)': [91.67, 96.30, 95.07],
    'Eficiencia (%)': [35.44, 34.8, 34.85],
    'Mes Pico': ['March', 'July', 'July'],
    'Motivo': ['Medical consultation', 'Medical consultation', 'Medical consultation']
}

df_evolucion = pd.DataFrame(data)
promedio_hijos = RRHH_full[RRHH_full['Son'] > 0]['Absenteeism_hours'].mean()
promedio_sin_hijos = RRHH_full[RRHH_full['Son']
                               == 0]['Absenteeism_hours'].mean()
diff_familiar = promedio_hijos - promedio_sin_hijos

costo_km_promedio = (RRHH_full['Transportation_expense'] /
                     RRHH_full['Distance_Residence_Work']).mean()

horas_bebedores = RRHH_full[RRHH_full['Social_drinker']
                            == 1]['Absenteeism_hours'].sum()
total_horas_abs = RRHH_full['Absenteeism_hours'].sum()
pct_horas_social = (horas_bebedores / total_horas_abs) * 100


st.header("Evolución de Gestión de Absentismo")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Ausencia Actual", "95.07%")

with kpi2:
    st.metric("Eficiencia Relativa", "34.85%",
              delta="Sin cambios", delta_color="off")

with kpi3:
    st.metric("Motivo Principal", "Medical Consultation",
              help="Medical consultation es el motivo recurrente")

with kpi4:
    st.metric(
        label="Factor Familiar",
        value=f"+{diff_familiar:.1f} horas",
        delta="Exceso de ausencia (Padres)",
        delta_color="inverse",
        help="Diferencia de horas de ausencia promedio entre empleados con hijos vs. sin hijos."
    )

with kpi5:
    st.metric(
        label="Impacto Hábitos Sociales",
        value=f"{pct_horas_social:.1f}%",
        delta="Social Drinkers",
        delta_color="normal",
        help="Porcentaje del total de horas de ausencia que corresponden a bebedores sociales."
    )
fig_evolucion = px.line(
    df_evolucion,
    x='Semana',
    y=['Ausencia (%)', 'Eficiencia (%)'],
    markers=True,
    title="<b>Tendencia: Ausencia vs Eficiencia</b>",
    labels={"value": "Porcentaje (%)", "variable": "Indicador"},
    color_discrete_map={
        "Ausencia (%)": "#E74C3C",
        "Eficiencia (%)": "#3498DB"
    }
)


fig_evolucion.update_traces(
    textposition="top center",
    texttemplate='%{y:.2f}%',
    mode="lines+markers+text"
)

fig_evolucion.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="grey",
    # Título
    title=dict(
        font=dict(size=17),
        x=0,
        xanchor='left'
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=1,
        title_text="",
        font=dict(size=14)
    ),

    hovermode="x unified",
    margin=dict(t=100, b=50)
)

fig_evolucion.update_xaxes(
    title_font=dict(size=20, color="grey"),
    tickfont=dict(size=16, color="grey"),
    gridcolor="rgba(200, 200, 200, 0.1)"
)

fig_evolucion.update_yaxes(
    title_font=dict(size=20, color="grey"),
    tickfont=dict(size=16, color="grey"),
    gridcolor="rgba(200, 200, 200, 0.1)",
    range=[0, 105]
)
st.plotly_chart(fig_evolucion, width='stretch',
                key="linea_evolucion_final")

st.divider()

st.header("Distribución de Rendimiento por Perfil")

df_empleados['Perfil_Riesgo'] = df_empleados['Perfil_Riesgo'].replace({
    'Junior Vulnerable': 'En desarrollo operativo'
})

total_n = len(df_empleados)
conteo = df_empleados['Perfil_Riesgo'].value_counts()


def fmt_manual(perfil):
    n = conteo.get(perfil, 0)
    pct = (n / total_n) * 100 if total_n > 0 else 0
    return f"{n} ({pct:.1f}%)"


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Perfil Estable",
    fmt_manual("Perfil Estable")
)

col2.metric(
    "Saturados Logísticos",
    fmt_manual("Saturado Logístico")
)

col3.metric(
    "En desarrollo operativo",
    fmt_manual("En desarrollo operativo")
)

col4.metric(
    "Perfil de Inestabilidad",
    fmt_manual("Perfil de Inestabilidad")
)

df_empleados['Perfil_Riesgo'] = df_empleados['Perfil_Riesgo'].replace({
    'Junior Vulnerable': 'En desarrollo operativo'
})

orden_perfiles = [
    'Perfil Estable',
    'En desarrollo operativo',
    'Saturado Logístico',
    'Perfil de Inestabilidad'
]

fig_dispersion = px.box(
    df_empleados,
    x='Perfil_Riesgo',
    y='Hit_target',
    color='Perfil_Riesgo',
    points="all",
    category_orders={"Perfil_Riesgo": orden_perfiles},
    labels={
        "Perfil_Riesgo": "Perfil Detectado",
        "Hit_target": "Rendimiento (%)"
    },
    color_discrete_map={
        'Perfil Estable': "#3A93D4",
        'Saturado Logístico': '#F1C40F',
        'En desarrollo operativo': "#DB3434",
        'Perfil de Inestabilidad': "#17A11E"
    },
    title="<b>Validación Visual: Distribución de Rendimiento por Perfil</b>"
)

fig_dispersion.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark",
    font_color="black",
    height=500,
    showlegend=False,
    title=dict(font=dict(size=18)),
    xaxis=dict(
        title=dict(font=dict(size=18)),
        tickfont=dict(size=16),
        categoryorder='array',
        categoryarray=orden_perfiles
    ),
    yaxis=dict(
        title=dict(font=dict(size=18)),
        tickfont=dict(size=18),
        gridcolor="rgba(0,0,0,0.1)"
    )

)

fig_dispersion.update_traces(marker=dict(line=dict(width=0)))


st.plotly_chart(fig_dispersion, width='stretch', key="box_final_ok")


st.markdown(f"""

        <p style="font-size: 13px; color: grey; line-height: 1.4;">
            Este análisis cruza las variables de <b>Antigüedad</b> (Service_time), <b>Logística</b> (Distance_Residence_Work), <b>Carga Laboral (Work_Load_Average) y  <b>Targes (Hit_Target).</b>
            <br>
        </p>

    """, unsafe_allow_html=True)

# ==========================================================
#  DETALLE ESTADÍSTICO: CLASIFICACIÓN MANUAL
# ==========================================================
with st.expander("Ver Reglas de Negocio"):
    st.write("Resumen de métricas basado en la lógica de clasificación original:")

    df_detalle_manual = df_empleados.groupby('Perfil_Riesgo').agg({
        'ID': 'count',
        'Work_load_Average_day': 'mean',
        'Distance_Residence_Work': 'mean',
        'Service_time': 'mean',
        'Hit_target': 'mean'
    }).rename(columns={'ID': 'Nº Empleados'})

    df_manual_fmt = df_detalle_manual.style.format({
        'Work_load_Average_day': '{:.1f} unidades',
        'Distance_Residence_Work': '{:.1f} km',
        'Service_time': '{:.1f} años',
        'Hit_target': '{:.1f}%'
    }).background_gradient(cmap='Blues', subset=['Hit_target'])

    st.dataframe(df_manual_fmt, width='stretch')

    st.caption(
        "Nota: Estos valores reflejan la situación actual bajo nuestros criterios.")


with st.expander("Ver Validación Estadística (ANOVA)"):

    grupo_ligera = df_analisis[df_analisis['Riesgo_Carga']
                               == 'Ligera']['Hit_target']
    grupo_optima = df_analisis[df_analisis['Riesgo_Carga']
                               == 'Óptima']['Hit_target']
    grupo_saturacion = df_analisis[df_analisis['Riesgo_Carga']
                                   == 'Saturación']['Hit_target']

    f_stat, p_val = stats.f_oneway(
        grupo_ligera, grupo_optima, grupo_saturacion)

    st.write(f"**Análisis de Varianza (Carga Laboral):**")
    st.write(f"- F-Statistic: `{f_stat:.4f}`")
    st.write(f"- P-Valor: `{p_val:.4f}`")

    if p_val < 0.05:
        st.success(
            "✅ Diferencias significativas confirmadas: La carga afecta el rendimiento.")
    else:
        st.warning("⚠️ No se detectaron diferencias significativas.")

st.info("""
**Lógica de Clasificación de Perfiles**

* **Perfil Estable:** Colaboradores que mantienen un equilibrio entre carga, distancia y experiencia.
* **En desarrollo operativo:** Empleados con menos de 5 años de antigüedad enfrentando cargas de trabajo superiores a la media.
* **Saturado Logístico:** Colaboradores con alta carga laboral (> 284 unidades) y que viven a más de 16km de la sede.
* **Perfil de Inestabilidad:** Empleados con fallos disciplinarios y carga de trabajo crítica (> 284 unidades).

*Nota: Los datos representan promedios agrupados por ID de empleado para capturar tendencias de largo plazo.*
""")

st.divider()

X = df_empleados[['Service_time',
                  'Distance_Residence_Work', 'Work_load_Average_day']]
y = df_empleados['Perfil_Riesgo']

clf = DecisionTreeClassifier(
    max_depth=3, class_weight='balanced', random_state=42)
clf.fit(X, y)

df_resumen = df_empleados.groupby('Perfil_Riesgo').agg({
    'ID': 'count',
    'Work_load_Average_day': 'mean',
    'Distance_Residence_Work': 'mean',
    'Hit_target': 'mean',
    'Service_time': 'mean'
}).rename(columns={'ID': 'Cantidad'}).reset_index()

# ==========================================================
#  BLOQUE DE INTELIGENCIA ARTIFICIAL (CLUSTERING)
# ==========================================================

columnas_ia = ['Service_time',
               'Distance_Residence_Work', 'Work_load_Average_day']
X_ia = df_empleados[columnas_ia]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ia)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_empleados['Cluster_Num'] = kmeans.fit_predict(X_scaled)

# ==========================================================
# 🧠 MAPEADOR CORREGIDO (Basado en datos reales de Imagen 3)
# ==========================================================

resumen_ia = df_empleados.groupby('Cluster_Num').agg({
    'Service_time': 'mean',
    'Distance_Residence_Work': 'mean',
    'Work_load_Average_day': 'mean'
})

id_junior = resumen_ia['Service_time'].idxmin()
id_logistico = resumen_ia['Distance_Residence_Work'].idxmax()
id_inestable = resumen_ia['Work_load_Average_day'].idxmax()

todos = set(resumen_ia.index)
usados = {id_junior, id_logistico, id_inestable}
id_estable = list(todos - usados)[0] if len(todos - usados) > 0 else None


mapeo_dinamico = {
    str(id_junior): "En desarrollo operativo",
    str(id_logistico): "Saturado Logístico",
    str(id_inestable): "Perfil de Inestabilidad",
    str(id_estable): "Perfil Estable"
}

df_empleados['Perfil_Riesgo'] = df_empleados['Cluster_Num'].astype(
    str).map(mapeo_dinamico)

df_resumen = df_empleados.groupby('Perfil_Riesgo').agg({
    'ID': 'count',
    'Work_load_Average_day': 'mean',
    'Distance_Residence_Work': 'mean',
    'Hit_target': 'mean',
    'Service_time': 'mean'
}).rename(columns={'ID': 'Cantidad'}).reset_index()


st.subheader("Resumen Operativo por Segmento (Validado por IA)")

total_empleados = len(df_empleados)
counts = df_empleados['Perfil_Riesgo'].value_counts()

m1, m2, m3, m4 = st.columns(4)


def fmt_metric(label):
    count = counts.get(label, 0)
    pct = (count / total_empleados) * 100
    return f"{count} ({pct:.1f}%)"


m1.metric(
    "Perfil Estable",
    fmt_metric("Perfil Estable"),
    "Grupo Base"
)

m2.metric(
    "Saturado Logístico",
    fmt_metric("Saturado Logístico"),
    "Riesgo Carga",
    delta_color="inverse"
)

m3.metric(
    "En desarrollo operativo",
    fmt_metric("En desarrollo operativo"),
    "En Formación"
)

m4.metric(
    "Perfil de Inestabilidad",
    fmt_metric("Perfil de Inestabilidad"),
    "Críticos",
    delta_color="inverse"
)

df_jitter = df_empleados.copy()
df_jitter['Distancia en Km'] = df_jitter['Distance_Residence_Work'] + \
    np.random.uniform(-0.3, 0.3, len(df_jitter))
df_jitter['Antigüedad'] = df_jitter['Service_time'] + \
    np.random.uniform(-0.3, 0.3, len(df_jitter))
fig_scatter = px.scatter(
    df_jitter,
    x='Distancia en Km',
    y='Antigüedad',
    color='Perfil_Riesgo',
    size='Hit_target',
    facet_col='Riesgo_Carga',
    template='plotly_white',
    size_max=12,
    color_discrete_map={
        'Perfil Estable': "#3A93D4",
        'Saturado Logístico': '#F1C40F',
        'En desarrollo operativo': "#DB3434",
        'Perfil de Inestabilidad': "#17A11E"
    }
)


fig_scatter.update_layout(
    paper_bgcolor="#EEEEEE",
    plot_bgcolor="#EEEEEE",
    font_color="#222222",
    margin=dict(t=100, b=100),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=1,
        title_text=""
    )
)

fig_scatter.update_xaxes(
    title_font=dict(size=18, color="#444444"),
    tickfont=dict(size=14, color="#444444"),
    gridcolor="white",
    linecolor="#444444",
    matches=None,
    showticklabels=True
)

fig_scatter.update_yaxes(
    title_font=dict(size=18, color="#444444"),
    tickfont=dict(size=14, color="#444444"),
    gridcolor="white",
    linecolor="#444444",
    matches=None,
    showticklabels=True
)

fig_scatter.for_each_annotation(lambda a: a.update(
    text=f"<b>{a.text.split('=')[-1]}</b>",
    font=dict(size=16, color="#222222")
))

fig_scatter.update_traces(marker=dict(line=dict(width=0), opacity=0.8))


st.plotly_chart(fig_scatter, width='stretch', key="scatter_claro")

with st.expander("Ver Umbrales de Carga detectados por el Modelo"):

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
        **Carga Óptima**
        **261 u. (Promedio)**
        *Punto de máxima eficiencia operativa sin riesgo de burnout.*
        """)
    with c2:
        st.info("""
        **Carga Ligera**
        **Hasta 250 u.**
        *Perfil en formación o baja demanda. Espacio para crecimiento.*
        """)

    with c3:
        st.warning("""
        **Saturación Crítica**
        **> 338 u.**
        *Riesgo inminente de fatiga, errores y baja en el rendimiento.*
        """)

# ==========================================================
# DETALLE ESTADÍSTICO DE LOS CLUSTERS
# ==========================================================
with st.expander("Ver Caracterización Detallada de Perfiles (Medias IA)"):
    st.write("A continuación se presentan los valores promedio que definen matemáticamente a cada grupo identificado por la IA:")
    df_detalle_clusters = df_empleados.groupby('Perfil_Riesgo').agg({
        'ID': 'count',
        'Work_load_Average_day': 'mean',
        'Distance_Residence_Work': 'mean',
        'Service_time': 'mean',
        'Hit_target': 'mean'
    }).rename(columns={'ID': 'Nº Empleados'})

    df_formatted = df_detalle_clusters.style.format({
        'Work_load_Average_day': '{:.1f} unidades',
        'Distance_Residence_Work': '{:.1f} km',
        'Service_time': '{:.1f} años',
        'Hit_target': '{:.1f}%'
    }).background_gradient(cmap='Greens', subset=['Hit_target'])

    st.dataframe(df_formatted, width='stretch')

    st.info("""
    **Interpretación Técnica:**
    * **Nº Empleados:** Tamaño del segmento.
    * **Carga Laboral:** Esfuerzo promedio diario del grupo.
    * **Distancia:** Fatiga logística promedio.
    * **Antigüedad:** Nivel de experiencia del grupo.
    * **Hit Target:** Rendimiento real alcanzado por cada perfil.
    """)
st.markdown(
    "<hr><p style='text-align: center;'>© 2026 | Desarrollado por José, Laura, Dani y Cristina</p>",
    unsafe_allow_html=True
)
