import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import subprocess
import sys

st.set_page_config(
    page_title="ML Monitoring Dashboard",
    layout="wide"
)

# Titre
st.title("Credit Risk ML Monitoring Dashboard")
st.markdown("---")


@st.cache_data(ttl=60) 
def load_data_from_db(db_path="logs.db"):
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM prediction_logs ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Convertir timestamp en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()


def format_metric(value, format_type="number"):
    if format_type == "number":
        return f"{value:,.0f}"
    elif format_type == "percent":
        return f"{value:.2f}%"
    elif format_type == "ms":
        return f"{value:.2f} ms"
    elif format_type == "score":
        return f"{value:.4f}"
    return str(value)


# Charger les données
df = load_data_from_db()

if df.empty:
    st.warning("Aucune donnée disponible. Effectuez des prédictions via l'API pour voir les statistiques.")
    st.stop()

# Sidebar - Filtres
st.sidebar.header("Filtres")

# Filtre de temps
time_range = st.sidebar.selectbox(
    "Période",
    ["Dernière heure", "Dernières 24h", "7 derniers jours", "30 derniers jours", "Tout"],
    index=1
)

# Appliquer le filtre de temps
now = datetime.now()
if time_range == "Dernière heure":
    df = df[df['timestamp'] >= now - timedelta(hours=1)]
elif time_range == "Dernières 24h":
    df = df[df['timestamp'] >= now - timedelta(days=1)]
elif time_range == "7 derniers jours":
    df = df[df['timestamp'] >= now - timedelta(days=7)]
elif time_range == "30 derniers jours":
    df = df[df['timestamp'] >= now - timedelta(days=30)]

# Filtre sur le statut
status_filter = st.sidebar.multiselect(
    "Statut HTTP",
    options=df['status_code'].unique(),
    default=df['status_code'].unique()
)
df = df[df['status_code'].isin(status_filter)]

st.sidebar.markdown(f"**Total prédictions:** {len(df)}")

st.header("Métriques Clés")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_predictions = len(df)
    st.metric("Total Prédictions", format_metric(total_predictions))

with col2:
    avg_latency = df['latency_ms'].mean()
    st.metric("Latence Moyenne", format_metric(avg_latency, "ms"))

with col3:
    avg_inference = df['inference_time_ms'].mean()
    st.metric("Inférence Moyenne", format_metric(avg_inference, "ms"))

with col4:
    error_rate = (df['status_code'] != 200).sum() / len(df) * 100 if len(df) > 0 else 0
    st.metric("Taux d'Erreur", format_metric(error_rate, "percent"))

with col5:
    avg_score = df['prediction_score'].mean()
    st.metric("Score Moyen", format_metric(avg_score, "score"))

st.markdown("---")

st.header("Monitoring API")

col1, col2 = st.columns(2)

with col1:
    # Graphique de latence
    st.subheader("Latence dans le temps")
    fig_latency = px.line(
        df.sort_values('timestamp'),
        x='timestamp',
        y='latency_ms',
        title="Latence totale (ms)"
    )
    fig_latency.add_hline(
        y=df['latency_ms'].quantile(0.95),
        line_dash="dash",
        line_color="red",
        annotation_text="P95"
    )
    st.plotly_chart(fig_latency, use_container_width=True)

with col2:
    # Distribution de la latence
    st.subheader("Distribution de la latence")
    fig_latency_dist = px.histogram(
        df,
        x='latency_ms',
        nbins=30,
        title="Distribution de la latence"
    )
    st.plotly_chart(fig_latency_dist, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Volume de requêtes par heure
    st.subheader("Volume de requêtes")
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly['timestamp'].dt.floor('H')
    volume_by_hour = df_hourly.groupby('hour').size().reset_index(name='count')

    fig_volume = px.bar(
        volume_by_hour,
        x='hour',
        y='count',
        title="Requêtes par heure"
    )
    st.plotly_chart(fig_volume, use_container_width=True)

with col4:
    # Taux d'erreur
    st.subheader("Statut des requêtes")
    status_counts = df['status_code'].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Distribution des codes HTTP"
    )
    st.plotly_chart(fig_status, use_container_width=True)

st.markdown("---")

st.header("Monitoring ML")

col1, col2 = st.columns(2)

with col1:
    # Distribution des scores
    st.subheader("Distribution des scores de prédiction")
    fig_scores = px.histogram(
        df,
        x='prediction_score',
        nbins=50,
        title="Distribution des scores (probabilité de défaut)"
    )
    fig_scores.add_vline(
        x=df['threshold'].iloc[0] if len(df) > 0 else 0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Seuil"
    )
    st.plotly_chart(fig_scores, use_container_width=True)

with col2:
    # Distribution des classes prédites
    st.subheader("Distribution des classes")
    class_counts = df['prediction_class'].value_counts()
    fig_classes = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        labels={'x': 'Classe', 'y': 'Nombre'},
        title="Prédictions par classe"
    )
    st.plotly_chart(fig_classes, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Évolution du score moyen dans le temps
    st.subheader("Évolution du score moyen")
    df_score_evolution = df.sort_values('timestamp').copy()
    df_score_evolution['hour'] = df_score_evolution['timestamp'].dt.floor('H')
    score_by_hour = df_score_evolution.groupby('hour')['prediction_score'].mean().reset_index()

    fig_score_evolution = px.line(
        score_by_hour,
        x='hour',
        y='prediction_score',
        title="Score moyen par heure"
    )
    st.plotly_chart(fig_score_evolution, use_container_width=True)

with col4:
    # Temps d'inférence
    st.subheader("Distribution du temps d'inférence")
    fig_inference = px.box(
        df,
        y='inference_time_ms',
        title="Temps d'inférence (ms)"
    )
    st.plotly_chart(fig_inference, use_container_width=True)

st.markdown("---")

st.header("Détails Techniques")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations du modèle")
    if len(df) > 0:
        st.write(f"**Version du modèle:** {df['model_version'].iloc[0]}")
        st.write(f"**Version de l'API:** {df['api_version'].iloc[0]}")
        st.write(f"**Environnement:** {df['environment'].iloc[0]}")
        st.write(f"**Nombre de features:** {df['n_features'].iloc[0]}")

with col2:
    st.subheader("Statistiques de performance")
    st.write(f"**P50 Latence:** {df['latency_ms'].quantile(0.5):.2f} ms")
    st.write(f"**P95 Latence:** {df['latency_ms'].quantile(0.95):.2f} ms")
    st.write(f"**P99 Latence:** {df['latency_ms'].quantile(0.99):.2f} ms")
    st.write(f"**Max Latence:** {df['latency_ms'].max():.2f} ms")

st.markdown("---")

st.header("Dernières Prédictions")

# Afficher les 10 dernières prédictions
display_columns = [
    'request_id', 'timestamp', 'prediction_score', 'prediction_class',
    'latency_ms', 'inference_time_ms', 'status_code'
]

st.dataframe(
    df[display_columns].head(10),
    use_container_width=True
)

# Bouton de rafraîchissement
if st.button("Rafraîchir les données"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")
st.markdown("*Dashboard mis à jour automatiquement toutes les 60 secondes*")

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
