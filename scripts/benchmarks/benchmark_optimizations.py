import time
import numpy as np
import pandas as pd
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class BenchmarkSuite:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {}

    def benchmark_latency(self, num_requests: int = 100):
        print(f"\nBenchmark Latence ({num_requests} requêtes)...")

        endpoint = f"{self.api_url}/api/v1/predict"
        sample_data = {
            "features": {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5
            }
        }

        latencies = []
        inference_times = []
        errors = 0

        for i in range(num_requests):
            start = time.time()
            try:
                response = requests.post(endpoint, json=sample_data, timeout=5)
                elapsed = (time.time() - start) * 1000  # en ms

                if response.status_code == 200:
                    latencies.append(elapsed)
                    data = response.json()
                    if 'inference_time_ms' in data:
                        inference_times.append(data['inference_time_ms'])
                else:
                    errors += 1

            except Exception as e:
                errors += 1

        if latencies:
            stats = {
                "total_requests": num_requests,
                "successful_requests": len(latencies),
                "errors": errors,
                "latency_mean": np.mean(latencies),
                "latency_median": np.median(latencies),
                "latency_p95": np.percentile(latencies, 95),
                "latency_p99": np.percentile(latencies, 99),
                "latency_min": np.min(latencies),
                "latency_max": np.max(latencies),
                "latency_std": np.std(latencies),
            }

            if inference_times:
                stats.update({
                    "inference_mean": np.mean(inference_times),
                    "inference_median": np.median(inference_times),
                    "overhead_mean": stats["latency_mean"] - np.mean(inference_times)
                })

            print(f"Moyenne: {stats['latency_mean']:.2f} ms")
            print(f"Médiane: {stats['latency_median']:.2f} ms")
            print(f"P95: {stats['latency_p95']:.2f} ms")
            print(f"P99: {stats['latency_p99']:.2f} ms")
            print(f"Taux d'erreur: {(errors/num_requests)*100:.2f}%")

            self.results['latency'] = stats
            return stats
        else:
            print("Aucune requête réussie")
            return None

    def benchmark_throughput(self, duration_seconds: int = 10):
        print(f"\nBenchmark Throughput ({duration_seconds}s)...")

        endpoint = f"{self.api_url}/api/v1/predict"
        sample_data = {
            "features": {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5
            }
        }

        start_time = time.time()
        requests_count = 0
        errors = 0

        while (time.time() - start_time) < duration_seconds:
            try:
                response = requests.post(endpoint, json=sample_data, timeout=2)
                if response.status_code == 200:
                    requests_count += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

        elapsed = time.time() - start_time
        throughput = requests_count / elapsed

        stats = {
            "duration_seconds": elapsed,
            "total_requests": requests_count,
            "errors": errors,
            "throughput_rps": throughput,
            "avg_time_per_request": (elapsed / requests_count * 1000) if requests_count > 0 else 0
        }

        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Requêtes réussies: {requests_count}")
        print(f"Erreurs: {errors}")

        self.results['throughput'] = stats
        return stats

    def benchmark_concurrent_load(self, concurrent_users: int = 10, requests_per_user: int = 10):
        print(f"\nBenchmark Charge Concurrente ({concurrent_users} utilisateurs, {requests_per_user} req/user)...")

        endpoint = f"{self.api_url}/api/v1/predict"
        sample_data = {
            "features": {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5
            }
        }

        def user_requests(user_id):
            latencies = []
            for _ in range(requests_per_user):
                start = time.time()
                try:
                    response = requests.post(endpoint, json=sample_data, timeout=10)
                    elapsed = (time.time() - start) * 1000
                    if response.status_code == 200:
                        latencies.append(elapsed)
                except Exception:
                    pass
            return latencies

        # Lancer les requêtes concurrentes
        start_time = time.time()
        all_latencies = []

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_requests, i) for i in range(concurrent_users)]

            for future in as_completed(futures):
                all_latencies.extend(future.result())

        total_time = time.time() - start_time
        total_requests = concurrent_users * requests_per_user

        if all_latencies:
            stats = {
                "concurrent_users": concurrent_users,
                "requests_per_user": requests_per_user,
                "total_requests": total_requests,
                "successful_requests": len(all_latencies),
                "total_time_seconds": total_time,
                "throughput_rps": len(all_latencies) / total_time,
                "latency_mean": np.mean(all_latencies),
                "latency_p95": np.percentile(all_latencies, 95),
                "latency_p99": np.percentile(all_latencies, 99)
            }

            print(f"Requêtes réussies: {len(all_latencies)}/{total_requests}")
            print(f"Throughput: {stats['throughput_rps']:.2f} req/s")
            print(f"Latence moyenne: {stats['latency_mean']:.2f} ms")
            print(f"Latence P95: {stats['latency_p95']:.2f} ms")

            self.results['concurrent_load'] = stats
            return stats
        else:
            print("Aucune requête réussie")
            return None

    def compare_with_baseline(self, baseline_file: str = "benchmark_baseline.json"):
        print(f"\nComparaison avec la baseline...")

        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)

            if 'latency' in self.results and 'latency' in baseline:
                current = self.results['latency']['latency_mean']
                base = baseline['latency']['latency_mean']
                improvement = ((base - current) / base) * 100

                print(f"\nAmélioration de la latence:")
                print(f"     Baseline: {base:.2f} ms")
                print(f"     Actuel: {current:.2f} ms")
                print(f"     Gain: {improvement:+.2f}%")

            if 'throughput' in self.results and 'throughput' in baseline:
                current = self.results['throughput']['throughput_rps']
                base = baseline['throughput']['throughput_rps']
                improvement = ((current - base) / base) * 100

                print(f"\nAmélioration du throughput:")
                print(f"     Baseline: {base:.2f} req/s")
                print(f"     Actuel: {current:.2f} req/s")
                print(f"     Gain: {improvement:+.2f}%")

        except FileNotFoundError:
            print(f"Fichier baseline non trouvé: {baseline_file}")
            print(f"Sauvegardez les résultats actuels comme baseline")

    def save_results(self, filename: str = None):
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nRésultats sauvegardés dans: {filename}")
        return filename

    def generate_report(self):
        print("\nGénération du rapport visuel...")

        if not self.results:
            print("Aucun résultat à afficher")
            return

        # Créer les graphiques
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Graphique 1: Latence
        if 'latency' in self.results:
            stats = self.results['latency']
            metrics = ['mean', 'median', 'p95', 'p99']
            values = [
                stats['latency_mean'],
                stats['latency_median'],
                stats['latency_p95'],
                stats['latency_p99']
            ]

            axes[0].bar(metrics, values, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
            axes[0].set_title('Latence de l\'API (ms)', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Temps (ms)')
            axes[0].grid(axis='y', alpha=0.3)

            for i, v in enumerate(values):
                axes[0].text(i, v + max(values)*0.02, f'{v:.1f}', ha='center', fontweight='bold')

        # Graphique 2: Throughput
        if 'throughput' in self.results:
            stats = self.results['throughput']
            axes[1].bar(['Throughput'], [stats['throughput_rps']], color='#9b59b6', width=0.5)
            axes[1].set_title('Throughput de l\'API', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Requêtes par seconde')
            axes[1].grid(axis='y', alpha=0.3)

            axes[1].text(0, stats['throughput_rps'] + stats['throughput_rps']*0.02,
                        f"{stats['throughput_rps']:.1f} req/s", ha='center', fontweight='bold')

        plt.tight_layout()
        filename = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Rapport visuel sauvegardé: {filename}")

        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark des optimisations de l'API")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="URL de l'API")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Nombre de requêtes pour le test de latence")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Sauvegarder comme baseline")
    parser.add_argument("--compare", action="store_true",
                        help="Comparer avec la baseline")

    args = parser.parse_args()

    benchmark = BenchmarkSuite(api_url=args.api_url)

    # Exécuter les benchmarks
    try:
        benchmark.benchmark_latency(num_requests=args.num_requests)
        benchmark.benchmark_throughput(duration_seconds=5)
        benchmark.benchmark_concurrent_load(concurrent_users=5, requests_per_user=10)

        # Sauvegarder les résultats
        filename = "benchmark_baseline.json" if args.save_baseline else None
        benchmark.save_results(filename)

        # Comparer avec baseline
        if args.compare and not args.save_baseline:
            benchmark.compare_with_baseline()

        # Générer le rapport
        benchmark.generate_report()

    except Exception as e:
        print(f"\nErreur lors du benchmark: {e}")
        print("   Assurez-vous que l'API est lancée sur", args.api_url)


if __name__ == "__main__":
    main()
