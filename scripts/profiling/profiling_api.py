"""
Script de profiling pour analyser les performances de l'API
Utilise cProfile pour identifier les goulots d'étranglement
"""

import cProfile
import pstats
import io
import requests
import time
from pstats import SortKey
import json


def profile_api_endpoint(url: str, data: dict, num_requests: int = 100):
    """
    Profile un endpoint de l'API

    Args:
        url: URL de l'endpoint
        data: Données à envoyer
        num_requests: Nombre de requêtes à effectuer
    """
    print(f"\n🔍 Profiling {url} avec {num_requests} requêtes...")

    def make_requests():
        """Fonction à profiler"""
        response_times = []
        for i in range(num_requests):
            start = time.time()
            try:
                response = requests.post(url, json=data)
                elapsed = (time.time() - start) * 1000  # en ms
                response_times.append(elapsed)
            except Exception as e:
                print(f"Erreur requête {i}: {e}")

        return response_times

    # Créer un profiler
    profiler = cProfile.Profile()

    # Profiler les requêtes
    profiler.enable()
    response_times = make_requests()
    profiler.disable()

    # Afficher les statistiques
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(20)  # Top 20 fonctions

    print(s.getvalue())

    # Statistiques des temps de réponse
    if response_times:
        print(f"\n📊 Statistiques des temps de réponse:")
        print(f"  - Moyenne: {sum(response_times)/len(response_times):.2f} ms")
        print(f"  - Minimum: {min(response_times):.2f} ms")
        print(f"  - Maximum: {max(response_times):.2f} ms")
        print(f"  - P95: {sorted(response_times)[int(len(response_times)*0.95)]:.2f} ms")
        print(f"  - P99: {sorted(response_times)[int(len(response_times)*0.99)]:.2f} ms")

    # Sauvegarder les stats dans un fichier
    with open('profiling_results.txt', 'w') as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats(sortby)
        ps.print_stats()

    print(f"\n✅ Résultats sauvegardés dans profiling_results.txt")

    return response_times


def profile_prediction_workflow():
    """
    Profile le workflow complet de prédiction
    Identifie les goulots d'étranglement dans le code
    """
    print("\n🔬 Profiling du workflow de prédiction...")

    # Simuler le workflow sans API (profiling du code interne)
    from api.models.model_loader import ModelLoader
    from api.models.schemas import PredictionRequest

    def simulation_predictions(n: int = 1000):
        """Simuler n prédictions"""
        loader = ModelLoader()

        # Charger le modèle
        loaded = loader.load_model_from_mlflow()
        if not loaded:
            print("❌ Modèle non chargé, simulation impossible")
            return

        # Faire des prédictions
        sample_data = {
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5
        }

        for _ in range(n):
            try:
                loader.predict(sample_data)
            except Exception as e:
                pass

    # Profiler
    profiler = cProfile.Profile()
    profiler.enable()
    simulation_predictions(1000)
    profiler.disable()

    # Afficher les stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)

    print(s.getvalue())

    # Sauvegarder
    with open('profiling_model_predictions.txt', 'w') as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats()

    print(f"\n✅ Résultats sauvegardés dans profiling_model_predictions.txt")


def analyze_bottlenecks():
    """
    Analyse les résultats de profiling pour identifier les goulots
    """
    print("\n🔍 Analyse des goulots d'étranglement...")

    try:
        with open('profiling_results.txt', 'r') as f:
            content = f.read()

        # Identifier les fonctions qui prennent le plus de temps
        print("\n⚠️ Fonctions les plus coûteuses (> 1% du temps total):")
        lines = content.split('\n')
        for line in lines[6:26]:  # Lignes avec les stats
            if line.strip() and not line.startswith('---'):
                print(f"  {line}")

        print("\n💡 Recommandations d'optimisation:")
        print("  1. Identifier les fonctions avec temps cumulé élevé")
        print("  2. Vérifier les appels répétitifs (ncalls)")
        print("  3. Optimiser les fonctions avec tottime élevé")
        print("  4. Envisager la mise en cache pour les calculs répétitifs")

    except FileNotFoundError:
        print("❌ Fichier profiling_results.txt non trouvé")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profiling de l'API MLOps")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1/predict",
                        help="URL de l'endpoint à profiler")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Nombre de requêtes pour le profiling")
    parser.add_argument("--mode", choices=["api", "model", "analyze", "all"], default="all",
                        help="Mode de profiling")

    args = parser.parse_args()

    print("=" * 70)
    print("🔬 PROFILING DE L'API MLOPS")
    print("=" * 70)

    sample_data = {
        "features": {
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5
        }
    }

    if args.mode in ["api", "all"]:
        try:
            profile_api_endpoint(args.api_url, sample_data, args.num_requests)
        except Exception as e:
            print(f"⚠️ Profiling API impossible: {e}")
            print("   (Assurez-vous que l'API est lancée)")

    if args.mode in ["model", "all"]:
        try:
            profile_prediction_workflow()
        except Exception as e:
            print(f"⚠️ Profiling modèle impossible: {e}")

    if args.mode in ["analyze", "all"]:
        analyze_bottlenecks()

    print("\n" + "=" * 70)
    print("✅ Profiling terminé!")
    print("=" * 70)
