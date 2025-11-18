import shutil
import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd, shell=False, env=None):
    """Run a shell command and print output in real time."""
    print(f"\n>>> Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell, env=env, check=True)
    return result


def main():
    base_dir = Path.cwd()
    venv_dir = base_dir / "myvenv"

    # 1. Create virtual environment
    run_command(["python3.9", "-m", "venv", str(venv_dir)])
    
    # 2. Path to the virtual environment's Python
    venv_python = venv_dir / "bin" / "python"
    venv_pip = venv_dir / "bin" / "pip"

    # 3. Install requirements
    run_command([str(venv_pip), "install", "-r", "requirements.txt"])

    # 4. Copy data
    """Run a shell command and print output in real time."""
    print("\n>>> cp -r ~/Downloads/Sepsis/files/challenge-2019/1.0.0/training/training* ./inputs/")
    input_dir = Path.cwd() / "inputs"
    shutil.rmtree(input_dir)
    shutil.copytree("/home/mnawawy/Downloads/Sepsis/files/challenge-2019/1.0.0/training", input_dir, dirs_exist_ok=True)

    # 5. Change directory to URET
    os.chdir("URET")

    # 6. Install URET package in editable mode
    run_command([str(venv_pip), "install", "-e", "."])

if __name__ == "__main__":
    main()




