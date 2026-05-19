import subprocess
import sys

def install_packages():
    # Lista pakietów do wymuszenia konkretnych wersji
    packages = [
        "scikit-learn==1.8.0",
        "tensorflow>=2.16.1",
        "joblib"
    ]
    
    print("Rozpoczynam aktualizację środowiska...")
    print("-" * 40)
    
    for package in packages:
        try:
            print(f"Instalowanie/Aktualizacja: {package}...")
            # Wywołanie komendy pip install
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Sukces: {package} został zainstalowany.\n")
        except subprocess.CalledProcessError as e:
            print(f"Błąd podczas instalacji {package}: {e}\n")
    
    print("-" * 40)
    print("Proces zakończony. Spróbuj teraz uruchomić swój skrypt walidacyjny.")

if __name__ == "__main__":
    install_packages()