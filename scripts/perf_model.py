import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def amdahl_speedup(serial_fraction, num_nodes):
    return 1 / (serial_fraction + (1 - serial_fraction) / num_nodes)

def gustafson_speedup(serial_fraction, num_nodes):
    return num_nodes - serial_fraction * (num_nodes - 1)

def main():
    parser = argparse.ArgumentParser(description="Generate Amdahl and Gustafson speedup plots.")
    parser.add_argument("--serial-fraction", type=float, default=0.05, help="Serial fraction (MTGS control overhead).")
    parser.add_argument("--failure-rate", type=float, default=0.01, help="Failure rate probability.")
    parser.add_argument("--max-nodes", type=int, default=8, help="Maximum number of nodes to simulate.")
    args = parser.parse_args()

    nodes = np.arange(1, args.max_nodes + 1)
    
    # Baseline assumes perfect scaling without failures
    baseline_amdahl = np.array([amdahl_speedup(0, n) for n in nodes])
    baseline_gustafson = np.array([gustafson_speedup(0, n) for n in nodes])

    # MTGS incorporates the serial fraction (overhead)
    mtgs_amdahl = np.array([amdahl_speedup(args.serial_fraction, n) for n in nodes])
    mtgs_gustafson = np.array([gustafson_speedup(args.serial_fraction, n) for n in nodes])

    # Adjusting for failure rate linearly as an approximation for throughput degradation
    failure_penalty = 1 - args.failure_rate
    mtgs_amdahl_adjusted = mtgs_amdahl * failure_penalty
    mtgs_gustafson_adjusted = mtgs_gustafson * failure_penalty

    # Export CSV
    csv_file = "speedup_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nodes", "Baseline Amdahl", "MTGS Amdahl", "Baseline Gustafson", "MTGS Gustafson"])
        for i, n in enumerate(nodes):
            writer.writerow([n, baseline_amdahl[i], mtgs_amdahl_adjusted[i], baseline_gustafson[i], mtgs_gustafson_adjusted[i]])
    print(f"Results exported to {csv_file}")

    # Plot Amdahl
    plt.figure(figsize=(10, 5))
    plt.plot(nodes, baseline_amdahl, 'k--', label="Baseline (Ideal)")
    plt.plot(nodes, mtgs_amdahl_adjusted, 'b-o', label=f"MTGS (s={args.serial_fraction}, f={args.failure_rate})")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Amdahl Speedup")
    plt.title("Amdahl's Law Speedup (Fixed Problem Size)")
    plt.legend()
    plt.grid(True)
    plt.savefig("amdahl_speedup.pdf")
    plt.close()

    # Plot Gustafson
    plt.figure(figsize=(10, 5))
    plt.plot(nodes, baseline_gustafson, 'k--', label="Baseline (Ideal)")
    plt.plot(nodes, mtgs_gustafson_adjusted, 'r-s', label=f"MTGS (s={args.serial_fraction}, f={args.failure_rate})")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Gustafson Speedup")
    plt.title("Gustafson's Law Speedup (Scaled Problem Size)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gustafson_speedup.pdf")
    plt.close()

    print("Plots saved as amdahl_speedup.pdf and gustafson_speedup.pdf")

if __name__ == "__main__":
    main()
