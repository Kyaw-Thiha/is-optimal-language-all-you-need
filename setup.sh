#!/bin/bash

# This is a setup file tailored for running on vast.ai

set -e # Exit if any command fails

# === CONFIG ===
PROJECT_DIR="/workspace/is-optimal-language-all-you-need"
VENV_DIR="$PROJECT_DIR/.env"

# Usage: ./setup.sh 50629
SSH_PORT=$1
if [ -z "$SSH_PORT" ]; then
  SSH_PORT="<PORT>"
fi

REMOTE_IP=$(curl -s https://ipinfo.io/ip)

echo "📦 Installing Deadsnakes Python 3.11 and dependencies..."
apt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.11 python3.11-venv python3.11-dev

echo "📁 Setting up virtual environment at $VENV_DIR..."
cd "$PROJECT_DIR"
# if [[ -n "$VIRTUAL_ENV" ]]; then
#   deactivate
# fi
# deactivate
rm -rf .env
python3.11 -m venv .env

echo "⬆️ Installing pip..."
source .env/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip

echo "📦 Installing Python requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo "Creating the data folder..."
mkdir -p "$PROJECT_DIR/data"

echo ""
echo "📂 Ready. Now upload your data:"
echo "If you want to upload offline data, run this on your local machine:"
echo "  scp -P $SSH_PORT -r ~/Documents/Projects/is-optimal-language-all-you-need/data/preprocess root@$REMOTE_IP:$PROJECT_DIR/data/"
echo ""
echo "Run this to activate virtual environment:"
echo "  source .env/bin/activate"
echo ""
echo "✅ Setup complete!"
