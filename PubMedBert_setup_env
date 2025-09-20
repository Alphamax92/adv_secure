"""
Environment setup script for PubMedBERT adversarial detection training
Run this first in Google Colab to install all required dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False
    return True

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  GPU not available, will use CPU (training will be slower)")
    except ImportError:
        print("❌ PyTorch not installed yet")

def setup_environment():
    """Main setup function"""
    print("🚀 Setting up environment for BioBERT adversarial detection training...")
    print("=" * 70)

    # --- FIX for numpy.dtype size changed error ---
    print("Attempting to resolve numpy/pandas compatibility issues...")
    # Uninstall existing numpy and pandas to ensure clean reinstallation
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "pandas", "-y"], capture_output=True)
    print("Uninstalled existing numpy and pandas.")

    # Reinstall numpy and pandas first
    if not install_package("numpy>=1.21.0"): # Specify a compatible numpy version
        print("Failed to reinstall numpy. Please check your environment.")
        return
    if not install_package("pandas>=1.3.0"): # Specify a compatible pandas version
        print("Failed to reinstall pandas. Please check your environment.")
        return
    print("Reinstalled numpy and pandas.")
    # --- END FIX ---

    # List of remaining required packages
    packages = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "huggingface_hub>=0.10.0",
        "accelerate>=0.12.0",
        "tokenizers>=0.12.0",
        "captum>=0.6.0"
    ]

    print("\n📦 Installing remaining required packages...")
    print("-" * 40)

    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)

    print("\n" + "=" * 70)

    if failed_packages:
        print("❌ Some packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\nPlease try installing them manually or restart the runtime.")
    else:
        print("✅ All packages installed successfully!")

    print("\n🔍 Checking system configuration...")
    print("-" * 40)

    # Check GPU availability
    check_gpu()

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"🐍 Python version: {python_version}")

    # Create necessary directories
    directories = [
        "/content/biobert_telemedicine_model",
        "/content/logs",
        "/content/plots"
    ]

    print("\n📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")

    print("\n" + "=" * 70)
    print("🎉 Environment setup complete!")
    print("\nIMPORTANT: If you still encounter errors, please restart your Colab runtime (Runtime -> Restart runtime) and run the setup script again.")
    print("\nNext steps:")
    print("1. Upload your telemedicine_adversarial_dataset.csv to /content/")
    print("2. Run the training script: python /content/scripts/train_biobert.py")
    print("3. Check results in /content/biobert_telemedicine_model/")

    # Test imports
    print("\n🧪 Testing imports...")
    test_imports()

def test_imports():
    """Test if all required packages can be imported"""
    import_tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("captum", "Captum")
    ]

    for module, name in import_tests:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")

if __name__ == "__main__":
    setup_environment()
