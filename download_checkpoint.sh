#!/bin/bash

# Default RunPod SSH connection info
SSH_USER_HOST="1f5kuad6m8zprk-6441169a@ssh.runpod.io"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_PATH="/workspace/LamSonOCR/checkpoints/best_model.pt"

# Resolve local path relative to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOCAL_PATH="$SCRIPT_DIR/checkpoints/best_model.pt"

# Allow overriding via arguments
if [ ! -z "$1" ]; then
    SSH_USER_HOST=$1
fi

echo "📥 Downloading trained model checkpoint from RunPod..."
echo "🔗 Host: $SSH_USER_HOST"
echo "🔑 Key: $SSH_KEY"
echo "📂 Remote path: $REMOTE_PATH"
echo "📂 Local target: $LOCAL_PATH"
echo "============================================="

# Ensure local checkpoints directory exists
mkdir -p "$SCRIPT_DIR/checkpoints"

# Download using SCP with legacy option -O (required for RunPod SFTP subsystem limits)
scp -O -i "$SSH_KEY" "$SSH_USER_HOST:$REMOTE_PATH" "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    echo "============================================="
    echo "✅ Download complete! Model saved to: $LOCAL_PATH"
    ls -lh "$LOCAL_PATH"
else
    echo "❌ Error: Download failed."
    exit 1
fi
