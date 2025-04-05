import os
import subprocess
import sys
import platform

# Function to run a command and capture its output
def run_command(command):
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Safely decode the output, ignoring errors for invalid UTF-8 characters
    print(result.stdout.decode('utf-8', errors='ignore'))
    if result.stderr:
        print(result.stderr.decode('utf-8', errors='ignore'))

# Detect the Python command to use
if platform.system() == "Windows":
    python_cmd = "python"
else:
    python_cmd = "python3"

# Create virtual environment in .venv
print("Creating virtual environment...")
run_command(f"{python_cmd} -m venv .venv")

# Determine the activation script based on the OS
if platform.system() == "Windows":
    activate_script = ".venv\\Scripts\\activate"
else:
    activate_script = ".venv/bin/activate"

# Activate the virtual environment
if platform.system() == "Windows":
    print("Activating virtual environment (Windows)...")
    subprocess.call([activate_script], shell=True)
else:
    print("Activating virtual environment (macOS/Linux)...")
    subprocess.call(f"source {activate_script} && bash", shell=True)

# Upgrade pip
print("Upgrading pip...")
run_command(f"{python_cmd} -m pip install --upgrade pip")

# Install the Trainers module
print("Installing Trainers module...")
run_command(f"{python_cmd} -m pip install ./Trainers/")

# Install dependencies from requirements.txt
print("Installing dependencies from requirements.txt...")
run_command(f"{python_cmd} -m pip install -r requirements.txt")

print("Virtual environment setup complete.")
