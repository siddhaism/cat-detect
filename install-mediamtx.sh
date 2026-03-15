#!/usr/bin/env bash
# Downloads and installs MediaMTX for aarch64 (Raspberry Pi).
# Usage: bash install-mediamtx.sh [--service]
#   --service   Also install a systemd service for MediaMTX
set -euo pipefail

MEDIAMTX_VERSION="1.11.3"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/mediamtx"
INSTALL_SERVICE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service) INSTALL_SERVICE=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

ARCH=$(uname -m)
case "$ARCH" in
  aarch64) TARBALL="mediamtx_v${MEDIAMTX_VERSION}_linux_arm64v8.tar.gz";;
  x86_64)  TARBALL="mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz";;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1;;
esac

URL="https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VERSION}/${TARBALL}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading MediaMTX v${MEDIAMTX_VERSION} for ${ARCH}..."
curl -fSL "$URL" -o "$TMP_DIR/$TARBALL"

echo "Extracting..."
tar xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

echo "Installing mediamtx to $INSTALL_DIR..."
sudo install -m 755 "$TMP_DIR/mediamtx" "$INSTALL_DIR/mediamtx"

echo "Installing config to $CONFIG_DIR..."
sudo mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CONFIG_DIR/mediamtx.yml" ]]; then
  sudo cp "$TMP_DIR/mediamtx.yml" "$CONFIG_DIR/mediamtx.yml"
  echo "Default config installed. Edit $CONFIG_DIR/mediamtx.yml as needed."
else
  echo "Config already exists at $CONFIG_DIR/mediamtx.yml — skipping."
fi

if [[ "$INSTALL_SERVICE" -eq 1 ]]; then
  echo "Installing systemd service..."
  sudo tee /etc/systemd/system/mediamtx.service >/dev/null <<EOF
[Unit]
Description=MediaMTX RTSP/WebRTC Server
After=network.target

[Service]
ExecStart=$INSTALL_DIR/mediamtx $CONFIG_DIR/mediamtx.yml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable mediamtx
  echo "Service installed and enabled. Start with: sudo systemctl start mediamtx"
fi

echo ""
echo "MediaMTX v${MEDIAMTX_VERSION} installed successfully."
echo "  Binary:  $INSTALL_DIR/mediamtx"
echo "  Config:  $CONFIG_DIR/mediamtx.yml"
echo ""
echo "Quick test:  mediamtx"
echo "Then open:   http://<pi-ip>:8889/<stream-name>"
