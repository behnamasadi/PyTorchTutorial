#!/usr/bin/env python3
"""
Experiment runner for Brain Cancer MRI Classification
Runs multiple model configurations for comparison
"""

import subprocess
import sys
import time
from datetime import datetime


def run_experiment(model_name, epochs=5):
    """Run a single experiment with specified parameters"""

    print(f"\n{'='*60}")
    print(f"🚀 Starting experiment: {model_name}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 Batch size: Model-optimized")
    print(f"🔄 Epochs: {epochs}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run training with specified parameters
        cmd = [
            sys.executable, "train.py",
            "--model", model_name,
            "--epochs", str(epochs)
        ]

        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Experiment {model_name} completed successfully!")
        print(f"⏱️  Total time: {duration:.2f} seconds")

        return True, duration

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"❌ Experiment {model_name} failed!")
        print(f"⏱️  Time before failure: {duration:.2f} seconds")
        print(f"Error: {e}")

        return False, duration


def main():
    """Run experiments for different model architectures"""

    print("🧠 Brain Cancer MRI Classification - Model Comparison")
    print("=" * 60)

    # Define experiments with GPU-optimized settings
    experiments = [
        # Will use model-specific batch size (128)
        {"model": "resnet18", "epochs": 5},
        # Will use model-specific batch size (128)
        {"model": "efficientnet_b0", "epochs": 5},
        # Will use model-specific batch size (32)
        {"model": "swin_t", "epochs": 3},
        # Will use model-specific batch size (64)
        {"model": "resnet50", "epochs": 3},
    ]

    results = []
    total_start_time = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n🔬 Experiment {i}/{len(experiments)}")

        success, duration = run_experiment(
            model_name=exp["model"],
            epochs=exp["epochs"]
        )

        results.append({
            "model": exp["model"],
            "success": success,
            "duration": duration,
            **exp
        })

        # Wait a bit between experiments
        if i < len(experiments):
            print("⏸️  Waiting 10 seconds before next experiment...")
            time.sleep(10)

    total_duration = time.time() - total_start_time

    # Print summary
    print("\n" + "="*60)
    print("📊 EXPERIMENT SUMMARY")
    print("="*60)

    successful_experiments = 0
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{result['model']:15} | {status} | {result['duration']:8.2f}s | "
              f"Epochs: {result['epochs']} | Batch: Model-optimized")

        if result["success"]:
            successful_experiments += 1

    print("-" * 60)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {len(experiments) - successful_experiments}")
    print(
        f"Total time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")

    print("\n🔍 Check your monitoring dashboards:")
    print("📈 TensorBoard: tensorboard --logdir runs")
    print("📊 MLflow: mlflow ui")
    print("🔮 Wandb: https://wandb.ai")

    print("\n🎉 All experiments completed!")


if __name__ == "__main__":
    main()
