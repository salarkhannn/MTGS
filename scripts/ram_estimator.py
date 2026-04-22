import argparse
import csv

def compute_ram_requirements(base_param_bytes, nodes, opt_multiplier, safety_margin=1.2):
    # Base requirements per node
    # RAM needs space for model + optimizer state + shadow state
    # Actually, the shadow state is an exact copy of the parameters (cpu pinned)
    model_ram = base_param_bytes
    optimizer_ram = base_param_bytes * opt_multiplier
    shadow_ram = base_param_bytes
    
    total_node_ram_bytes = (model_ram + optimizer_ram + shadow_ram) * safety_margin
    return total_node_ram_bytes

def main():
    parser = argparse.ArgumentParser(description="Estimate CPU RAM for MTGS shadow states.")
    parser.add_argument("--params", type=int, default=66362880, help="Number of trainable parameters.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Precision format.")
    parser.add_argument("--opt-multiplier", type=float, default=2.0, help="Optimizer state multiplier (e.g., Adam=2).")
    parser.add_argument("--max-nodes", type=int, default=4, help="Maximum number of nodes to simulate.")
    args = parser.parse_args()

    bytes_per_param = 4 if args.precision == "fp32" else 2
    base_param_bytes = args.params * bytes_per_param

    results = []
    print(f"{'Nodes':<6} | {'Per-Node RAM (MB)':<18} | {'Cluster-wide RAM (MB)':<22} | {'Shadow RAM (MB)':<15}")
    print("-" * 65)

    with open("ram_estimates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nodes", "Per-Node RAM (MB)", "Cluster-wide RAM (MB)", "Shadow RAM (MB)"])
        
        for n in range(1, args.max_nodes + 1):
            per_node_bytes = compute_ram_requirements(base_param_bytes, n, args.opt_multiplier)
            per_node_mb = per_node_bytes / (1024 * 1024)
            cluster_mb = per_node_mb * n
            shadow_mb = base_param_bytes / (1024 * 1024)
            
            print(f"{n:<6} | {per_node_mb:<18.2f} | {cluster_mb:<22.2f} | {shadow_mb:<15.2f}")
            writer.writerow([n, per_node_mb, cluster_mb, shadow_mb])
    
    print("-" * 65)
    print("Estimates exported to ram_estimates.csv")
    print("Note: Values include a 20% safety margin.")

if __name__ == "__main__":
    main()
