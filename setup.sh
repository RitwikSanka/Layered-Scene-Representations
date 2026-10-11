#!/bin/bash
set -e

echo "Setting up Layered Representations Environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete. To activate: source venv/bin/activate"
