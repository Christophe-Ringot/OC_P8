import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


def analyze_api_performance(db_path: str = "logs.db", days: int = 7):
    conn = sqlite3.connect(db_path)

    # Date de début
    start_date = (datetime.now() - timedelta(days=days)).isoformat()

    # Récupérer les logs
    query = f"""
        SELECT
            timestamp,
            latency_ms,
            inference_time_ms,
            status_code,
            prediction_score,
            model_version
        FROM prediction_logs
        WHERE timestamp >= '{start_date}'
        ORDER BY timestamp
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(f"Aucune donnée disponible pour les {days} derniers jours")
        return

    print(f"ANALYSE DES PERFORMANCES - {days} derniers jours")


    # Statistiques globales
    print(f"  Nombre total de requêtes: {len(df)}")
    print(f"  Période: {df['timestamp'].min()} à {df['timestamp'].max()}")

    # Latence API
    print(f"  Moyenne: {df['latency_ms'].mean():.2f}")
    print(f"  Médiane (P50): {df['latency_ms'].median():.2f}")
    print(f"  P95: {df['latency_ms'].quantile(0.95):.2f}")
    print(f"  P99: {df['latency_ms'].quantile(0.99):.2f}")
    print(f"  Max: {df['latency_ms'].max():.2f}")
    print(f"  Min: {df['latency_ms'].min():.2f}")

    # Temps d'inférence
    print(f"  Moyenne: {df['inference_time_ms'].mean():.2f}")
    print(f"  Médiane (P50): {df['inference_time_ms'].median():.2f}")
    print(f"  P95: {df['inference_time_ms'].quantile(0.95):.2f}")
    print(f"  P99: {df['inference_time_ms'].quantile(0.99):.2f}")
    print(f"  Max: {df['inference_time_ms'].max():.2f}")

    # Taux d'erreur
    error_rate = (df['status_code'] != 200).sum() / len(df) * 100
    print(f"\nTaux d'erreur: {error_rate:.2f}%")

    if error_rate > 0:
        print("\nCodes de statut:")
        status_counts = df['status_code'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/len(df)*100:.1f}%)")

    # Distribution des scores de prédiction
    print(f"  Moyenne: {df['prediction_score'].mean():.3f}")
    print(f"  Médiane: {df['prediction_score'].median():.3f}")
    print(f"  Std: {df['prediction_score'].std():.3f}")

    # Analyse temporelle
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    hourly_counts = df['hour'].value_counts().sort_index()
    for hour, count in hourly_counts.items():
        print(f"  {hour:02d}h: {count} requêtes")

    # Goulots d'étranglement

    # Requêtes les plus lentes
    slow_threshold = df['latency_ms'].quantile(0.95)
    slow_requests = df[df['latency_ms'] > slow_threshold]

    if len(slow_requests) > 0:
        print(f"\n{len(slow_requests)} requêtes dépassent le P95 ({slow_threshold:.2f} ms)")
        print(f"  Latence moyenne de ces requêtes: {slow_requests['latency_ms'].mean():.2f} ms")

    # Comparaison latence vs temps d'inférence
    overhead_ms = df['latency_ms'].mean() - df['inference_time_ms'].mean()
    overhead_pct = (overhead_ms / df['latency_ms'].mean()) * 100

    print(f"\n  Overhead moyen (hors inférence): {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")


def analyze_model_performance(db_path: str = "logs.db"):
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            prediction_score,
            prediction_class,
            model_version,
            timestamp
        FROM prediction_logs
        WHERE status_code = 200
        ORDER BY timestamp DESC
        LIMIT 1000
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("Aucune donnée de prédiction disponible")
        return


    print(f"\nNombre de prédictions analysées: {len(df)}")
    print(f"Version du modèle: {df['model_version'].iloc[0] if not df.empty else 'N/A'}")

    # Distribution des classes prédites
    class_dist = df['prediction_class'].value_counts()
    for cls, count in class_dist.items():
        print(f"  Classe {cls}: {count} ({count/len(df)*100:.1f}%)")

    # Distribution des scores
    print(f"  Moyenne: {df['prediction_score'].mean():.3f}")
    print(f"  Médiane: {df['prediction_score'].median():.3f}")
    print(f"  Écart-type: {df['prediction_score'].std():.3f}")
    print(f"  Min: {df['prediction_score'].min():.3f}")
    print(f"  Max: {df['prediction_score'].max():.3f}")

    # Scores par décile
    for i in range(0, 11):
        q = i / 10
        val = df['prediction_score'].quantile(q)
        print(f"  P{i*10}: {val:.3f}")


if __name__ == "__main__":
    # Analyser les performances de l'API
    analyze_api_performance(days=7)

    # Analyser les performances du modèle
    analyze_model_performance()
