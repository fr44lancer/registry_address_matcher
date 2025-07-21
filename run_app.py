#!/usr/bin/env python3
"""
Streamlit app runner with proper path setup
"""
import os
import sys
import subprocess

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Change to the app directory
os.chdir(current_dir)

if __name__ == "__main__":
    # Run streamlit with the app.py file
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"] + sys.argv[1:])