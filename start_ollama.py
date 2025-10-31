import subprocess
import requests
import time
import sys
import os

def check_ollama_running():
    """Check if Ollama is already running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    print("🚀 Starting Ollama server...")
    
    # Check if already running
    if check_ollama_running():
        print("✅ Ollama is already running")
        return True
    
    print("🔧 Ollama not running, starting now...")
    
    try:
        # Start Ollama in the background
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup with timeout
        print("⏳ Waiting for Ollama to start...")
        for i in range(30):  # 30 second timeout
            time.sleep(1)
            if check_ollama_running():
                print("✅ Ollama started successfully!")
                return True
            if i % 5 == 0:
                print(f"   Still starting... {i+1}s")
        
        print("❌ Ollama failed to start within 30 seconds")
        return False
        
    except FileNotFoundError:
        print("❌ Ollama command not found. Please install Ollama first.")
        print("   Run: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return False

def main():
    if start_ollama():
        print("\n🎯 Now you can run your preprocessing:")
        print("   python3 1_preprocessData.py")
    else:
        print("\n💡 Please start Ollama manually:")
        print("   ollama serve")
        print("   Then in a NEW terminal, run your code.")
        sys.exit(1)

if __name__ == "__main__":
    main()