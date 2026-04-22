import argparse
import csv
import json
import sys
from transformers import AutoConfig, AutoModelForMaskedLM

def get_bytes_per_param(precision: str) -> int:
    if precision == "fp32":
        return 4
    elif precision in ["fp16", "bf16"]:
        return 2
    else:
        raise ValueError(f"Unknown precision: {precision}")

def profile_payload(model_name: str, precision: str, output_format: str):
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_config(config)
    except Exception as e:
        print(f"Error loading model config: {e}")
        sys.exit(1)

    bytes_per_param = get_bytes_per_param(precision)
    layer_stats = []
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            payload_bytes = num_params * bytes_per_param
            layer_stats.append({
                "layer_name": name,
                "shape": list(param.shape),
                "num_parameters": num_params,
                "payload_bytes": payload_bytes
            })

    total_payload_bytes = total_params * bytes_per_param
    total_payload_mb = total_payload_bytes / (1024 * 1024)

    summary = {
        "model_name": model_name,
        "precision": precision,
        "total_trainable_parameters": total_params,
        "total_payload_bytes": total_payload_bytes,
        "total_payload_mb": total_payload_mb,
        "layers": layer_stats
    }

    if output_format == "json":
        with open("payload_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Summary exported to payload_summary.json")
    elif output_format == "csv":
        with open("payload_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Layer Name", "Shape", "Num Parameters", "Payload (Bytes)"])
            for layer in layer_stats:
                writer.writerow([layer["layer_name"], layer["shape"], layer["num_parameters"], layer["payload_bytes"]])
            writer.writerow([])
            writer.writerow(["Total Trainable Params", total_params])
            writer.writerow(["Total Payload (Bytes)", total_payload_bytes])
            writer.writerow(["Total Payload (MB)", total_payload_mb])
        print("Summary exported to payload_summary.csv")
    else:
        print("Unsupported output format.")
    
    print(f"Model: {model_name} | Precision: {precision}")
    print(f"Total Params: {total_params:,}")
    print(f"Total Payload: {total_payload_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model gradient payload.")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Model name or path.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Precision format.")
    parser.add_argument("--format", type=str, choices=["csv", "json"], default="csv", help="Output format.")
    args = parser.parse_args()
    
    profile_payload(args.model, args.precision, args.format)
