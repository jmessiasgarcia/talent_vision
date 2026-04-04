import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_ia_pipeline(df_id):
    """
    Recibe el DataFrame agrupado por ID y realiza:
    1. Escalado de variables.
    2. Clustering KMeans (3 grupos).
    3. Mapeo dinámico de nombres.
    4. Reducción de dimensionalidad PCA.
    """
    # 1. Selección de features (las que tenías en tu código)
    features = ['Age', 'Body_mass_index', 'Son',
                'Education', 'Social_drinker', 'Social_smoker']
    X = df_id[features]

    # 2. Escalado
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

    # 3. KMeans (Estable)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_id['Cluster_ID'] = cluster_labels

    # 4. Mapeo Dinámico (Para que siempre asigne bien los nombres por edad)
    # Esto evita que si los datos cambian, los nombres se mezclen
    centros_edad = df_id.groupby('Cluster_ID')[
        'Age'].mean().sort_values().index
    mapeo_nombres = {
        centros_edad[0]: "Talento Enfocado",
        centros_edad[1]: "Motor Familiar",
        centros_edad[2]: "Talento Senior"
    }
    df_id['Segmento'] = df_id['Cluster_ID'].map(mapeo_nombres)

    # 5. PCA (Para visualización 2D)
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(X_scaled)
    df_id['PCA1'] = pca_data[:, 0]
    df_id['PCA2'] = pca_data[:, 1]

    # Añadimos un poco de jitter (ruido) para que los puntos no se solapen en el gráfico
    df_id["PCA1_jitter"] = df_id["PCA1"] + \
        np.random.uniform(-0.1, 0.1, len(df_id))
    df_id["PCA2_jitter"] = df_id["PCA2"] + \
        np.random.uniform(-0.1, 0.1, len(df_id))

    # 6. Métricas de Calidad
    sil_score = silhouette_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)

    return df_id, sil_score, db_score, X_scaled
