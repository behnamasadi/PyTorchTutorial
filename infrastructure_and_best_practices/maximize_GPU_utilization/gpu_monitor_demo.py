#!/usr/bin/env python3
"""
GPU Monitoring Demo using GPUtil
Shows how to monitor GPU usage in real-time
"""

import GPUtil
import psutil
import time
import torch
import numpy as np


def print_gpu_info():
    """Print detailed GPU information"""
    print("🚀 GPU Information:")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    try:
        gpus = GPUtil.getGPUs()

        if not gpus:
            print("❌ No GPUs found")
            return

        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            print(f"  📊 Utilization: {gpu.load * 100:.1f}%")
            print(
                f"  🧠 Memory: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f}MB ({gpu.memoryUtil * 100:.1f}%)")
            print(f"  🌡️  Temperature: {gpu.temperature}°C")
            print(f"  🆔 UUID: {gpu.uuid}")
            print(f"  🔧 Driver: {gpu.driver}")
            print()

    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")


def monitor_gpu_during_load():
    """Monitor GPU while creating artificial load"""
    print("🔥 Creating GPU load for monitoring demo...")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available for load test")
        return

    # Create some GPU load
    device = torch.device('cuda')

    # Create large tensors to use GPU memory and compute
    print("📦 Allocating GPU memory...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    print("⚡ Starting computation...")
    for i in range(10):
        # Do some computation to create GPU load
        z = torch.matmul(x, y)
        z = torch.relu(z)
        z = torch.softmax(z, dim=1)

        # Monitor GPU every iteration
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"Iteration {i+1:2d} | "
                  f"GPU: {gpu.load * 100:5.1f}% | "
                  f"Memory: {gpu.memoryUsed:4.0f}MB ({gpu.memoryUtil * 100:5.1f}%) | "
                  f"Temp: {gpu.temperature:3.0f}°C")

        time.sleep(1)

    print("✅ GPU load test completed!")


def continuous_monitoring(duration=30):
    """Monitor GPU continuously for specified duration"""
    print(f"📊 Continuous GPU monitoring for {duration} seconds...")
    print("=" * 50)
    print("Time     | GPU%   | Memory%  | Used/Total MB | Temp°C")
    print("-" * 50)

    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                elapsed = time.time() - start_time

                print(f"{elapsed:6.1f}s | "
                      f"{gpu.load * 100:5.1f}% | "
                      f"{gpu.memoryUtil * 100:7.1f}% | "
                      f"{gpu.memoryUsed:4.0f}/{gpu.memoryTotal:4.0f}MB | "
                      f"{gpu.temperature:5.0f}°C")
            else:
                print("No GPU found")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(2)


def get_available_memory():
    """Check available GPU memory before training"""
    if not torch.cuda.is_available():
        return 0

    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            available_mb = gpu.memoryTotal - gpu.memoryUsed
            return available_mb
    except:
        return 0


def main():
    """Main demo function"""
    print("🔍 GPUtil Demo - GPU Monitoring")
    print("=" * 60)

    # Basic GPU info
    print_gpu_info()

    # Check available memory
    available = get_available_memory()
    print(f"💾 Available GPU Memory: {available:.0f}MB")
    print()

    # Ask user what they want to do
    print("Choose a demo:")
    print("1. Monitor GPU during artificial load")
    print("2. Continuous monitoring (30 seconds)")
    print("3. Quick GPU status check")
    print("4. Exit")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        monitor_gpu_during_load()
    elif choice == "2":
        continuous_monitoring(30)
    elif choice == "3":
        print_gpu_info()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
