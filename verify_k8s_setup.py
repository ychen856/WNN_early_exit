#!/usr/bin/env python3
"""Verify Kubernetes setup for multi-worker data loading."""

import os
import sys
import subprocess

def check_env():
    """Check environment variables."""
    print("=" * 70)
    print("[1] ENVIRONMENT CHECK")
    print("=" * 70)
    
    in_docker = os.path.exists('/.dockerenv')
    in_k8s = 'KUBERNETES_SERVICE_HOST' in os.environ
    
    print(f"  Docker detected: {in_docker}")
    print(f"  Kubernetes detected: {in_k8s}")
    
    if in_k8s:
        print(f"  KUBERNETES_SERVICE_HOST: {os.environ.get('KUBERNETES_SERVICE_HOST')}")
        print(f"  KUBERNETES_SERVICE_PORT: {os.environ.get('KUBERNETES_SERVICE_PORT')}")
    
    return in_docker or in_k8s

def check_shm():
    """Check /dev/shm availability and size."""
    print("\n" + "=" * 70)
    print("[2] SHARED MEMORY CHECK")
    print("=" * 70)
    
    shm_path = '/dev/shm'
    
    if not os.path.exists(shm_path):
        print(f"  ERROR: {shm_path} does not exist!")
        return False
    
    if not os.access(shm_path, os.W_OK):
        print(f"  ERROR: {shm_path} is not writable!")
        return False
    
    try:
        import statvfs
        stat = os.statvfs(shm_path)
        total_bytes = stat.f_blocks * stat.f_frsize
        avail_bytes = stat.f_bavail * stat.f_frsize
        total_gb = total_bytes / (1024**3)
        avail_gb = avail_bytes / (1024**3)
        
        print(f"  {shm_path} status: OK")
        print(f"  Total size: {total_gb:.2f} GB")
        print(f"  Available: {avail_gb:.2f} GB")
        
        if total_gb < 4:
            print(f"  WARNING: Only {total_gb:.2f}GB allocated. Recommend 8GB+")
            return False
        
        return True
    except Exception as e:
        print(f"  ERROR: Could not check size: {e}")
        return False

def check_pytorch():
    """Check PyTorch and dataloader capabilities."""
    print("\n" + "=" * 70)
    print("[3] PYTORCH CHECK")
    print("=" * 70)
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"  ERROR: PyTorch not available: {e}")
        return False

def test_dataloader():
    """Test multi-worker DataLoader."""
    print("\n" + "=" * 70)
    print("[4] DATALOADER TEST (num_workers=4)")
    print("=" * 70)
    
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy dataset
        data = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, labels)
        
        # Try with 4 workers
        loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
        
        print("  Creating DataLoader with num_workers=4... OK")
        
        # Try to iterate
        for batch_idx, (data, labels) in enumerate(loader):
            if batch_idx == 0:
                print(f"  First batch loaded successfully: {data.shape}")
                break
        
        print("  Multi-worker DataLoader test: PASSED")
        return True
        
    except RuntimeError as e:
        if "shared memory" in str(e) or "shm" in str(e):
            print(f"  ERROR: Shared memory error: {e}")
            print()
            print("  FIX: Add SHM volume to pod spec:")
            print("    volumeMounts:")
            print("    - mountPath: /dev/shm")
            print("      name: shm")
            print("    volumes:")
            print("    - name: shm")
            print("      emptyDir:")
            print("        medium: Memory")
            print("        sizeLimit: 8Gi")
            return False
        else:
            print(f"  ERROR: {e}")
            return False
    except Exception as e:
        print(f"  ERROR: Unexpected error: {e}")
        return False

def main():
    print("\n" + "=" * 70)
    print("KUBERNETES SETUP VERIFICATION FOR PYTORCH DATALOADER")
    print("=" * 70 + "\n")
    
    results = {
        "env": check_env(),
        "shm": check_shm(),
        "pytorch": check_pytorch(),
        "dataloader": test_dataloader(),
    }
    
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check.upper():20s} {status}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n[OK] Your system is ready for multi-worker data loading!")
        print("     You can use --num-workers 4-8 safely")
        return 0
    else:
        print("\n[ERROR] Some checks failed. See details above.")
        print("        For Kubernetes, ensure pod spec includes SHM volume")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())
