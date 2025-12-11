#!/usr/bin/env python3
"""
Market Intelligence Platform - Quick Start Script
=================================================
Script para configurar e iniciar la plataforma rápidamente.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_banner():
    """Imprime banner de inicio."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🚀 MARKET INTELLIGENCE AI PLATFORM                        ║
    ║                                                              ║
    ║   Inteligencia de Mercado Confiable impulsada por IA        ║
    ║   RAG + Multi-Agent + Triple Verification                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Verifica la versión de Python."""
    if sys.version_info < (3, 10):
        print("❌ Error: Se requiere Python 3.10 o superior")
        print(f"   Versión actual: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")


def create_virtual_env():
    """Crea entorno virtual si no existe."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creando entorno virtual...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Entorno virtual creado")
    else:
        print("✅ Entorno virtual existente")
    return venv_path


def get_pip_path(venv_path):
    """Obtiene la ruta al pip del entorno virtual."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "pip"
    return venv_path / "bin" / "pip"


def install_dependencies(venv_path):
    """Instala dependencias."""
    pip_path = get_pip_path(venv_path)
    
    print("📥 Instalando dependencias...")
    print("   (Esto puede tomar varios minutos la primera vez)")
    
    # Actualizar pip
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Instalar requirements
    req_file = Path("backend/requirements.txt")
    if req_file.exists():
        subprocess.run([str(pip_path), "install", "-r", str(req_file)], check=True)
        print("✅ Dependencias instaladas")
    else:
        print("❌ No se encontró requirements.txt")
        sys.exit(1)


def setup_env_file():
    """Configura archivo .env."""
    env_example = Path("backend/.env.example")
    env_file = Path("backend/.env")
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("📝 Archivo .env creado desde .env.example")
        print("")
        print("⚠️  IMPORTANTE: Configura tus API keys en backend/.env")
        print("   - ANTHROPIC_API_KEY o OPENAI_API_KEY (obligatorio)")
        print("   - FINNHUB_API_KEY (opcional, para datos financieros)")
        print("")
    elif env_file.exists():
        print("✅ Archivo .env existente")
    else:
        print("⚠️  No se encontró .env.example")


def create_directories():
    """Crea directorios necesarios."""
    dirs = [
        "backend/data/chroma_db",
        "backend/logs",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directorios creados")


def check_api_keys():
    """Verifica si hay API keys configuradas."""
    env_file = Path("backend/.env")
    
    if env_file.exists():
        with open(env_file, "r") as f:
            content = f.read()
            
        has_anthropic = "ANTHROPIC_API_KEY=" in content and "your_" not in content.split("ANTHROPIC_API_KEY=")[1].split("\n")[0]
        has_openai = "OPENAI_API_KEY=" in content and "your_" not in content.split("OPENAI_API_KEY=")[1].split("\n")[0] if "OPENAI_API_KEY=" in content else False
        
        if has_anthropic or has_openai:
            print("✅ API key de LLM configurada")
            return True
        else:
            print("")
            print("⚠️  ADVERTENCIA: No se detectó API key de LLM configurada")
            print("   El sistema funcionará en modo demo sin generación de IA")
            print("   Configura ANTHROPIC_API_KEY o OPENAI_API_KEY en backend/.env")
            print("")
            return False
    return False


def run_server(venv_path):
    """Ejecuta el servidor."""
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python"
    else:
        python_path = venv_path / "bin" / "python"
    
    print("")
    print("🚀 Iniciando servidor...")
    print("")
    print("   API Docs:  http://localhost:8000/docs")
    print("   ReDoc:     http://localhost:8000/redoc")
    print("   Health:    http://localhost:8000/health")
    print("")
    print("   Presiona Ctrl+C para detener")
    print("")
    
    os.chdir("backend")
    subprocess.run([
        str(python_path), "-m", "uvicorn",
        "app.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])


def main():
    """Función principal."""
    print_banner()
    
    # Verificar Python
    check_python_version()
    
    # Setup
    venv_path = create_virtual_env()
    install_dependencies(venv_path)
    create_directories()
    setup_env_file()
    
    # Verificar API keys
    check_api_keys()
    
    # Preguntar si iniciar servidor
    print("")
    response = input("¿Iniciar el servidor ahora? [Y/n]: ").strip().lower()
    
    if response in ["", "y", "yes", "s", "si"]:
        run_server(venv_path)
    else:
        print("")
        print("Para iniciar el servidor manualmente:")
        print("  cd backend")
        print("  source ../venv/bin/activate  # Linux/Mac")
        print("  uvicorn app.main:app --reload")
        print("")


if __name__ == "__main__":
    main()
