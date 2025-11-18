#!/usr/bin/env bash
set -euo pipefail

echo "==> Detecting OS and CUDA ..."
if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  echo "OS: $NAME ($VERSION)"
else
  echo "Cannot detect OS release. Aborting."
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not found in PATH (that's okay; we only need the NVIDIA repo to install Nsight tools)."
fi

# Determine Ubuntu series (22.04 -> ubuntu2204, 24.04 -> ubuntu2404)
UBU_SERIES=""
case "${VERSION_ID:-}" in
  22.04) UBU_SERIES="ubuntu2204" ;;
  24.04) UBU_SERIES="ubuntu2404" ;;
  *)
    echo "Unsupported/untested Ubuntu version: ${VERSION_ID:-unknown}. Trying ubuntu2404 repo by default."
    UBU_SERIES="ubuntu2404"
    ;;
esac

echo "==> Adding NVIDIA CUDA apt repository via cuda-keyring (series: $UBU_SERIES)"
TMP_DEB="$(mktemp /tmp/cuda-keyring_XXXX.deb)"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBU_SERIES}/x86_64/cuda-keyring_1.1-1_all.deb"

echo "Downloading: $KEYRING_URL"
curl -fsSL "$KEYRING_URL" -o "$TMP_DEB"

echo "Installing keyring: $TMP_DEB"
sudo dpkg -i "$TMP_DEB" || true

echo "==> apt update"
sudo apt-get update -y

echo "==> Installing Nsight Systems and Nsight Compute ..."
set +e
sudo apt-get install -y nsight-systems nsight-compute
INSTALL_RC=$?
set -e

if [[ $INSTALL_RC -ne 0 ]]; then
  echo "First attempt failed (packages may be named/versioned differently). Showing availability diagnostics ..."
  echo "---- apt-cache policy nsight-systems ----"
  apt-cache policy nsight-systems || true
  echo "---- apt-cache policy nsight-compute ----"
  apt-cache policy nsight-compute || true
  echo "---- apt search nsight (truncated) ----"
  apt search nsight | sed -n '1,120p' || true
  echo
  echo "You may need to install a specific version, e.g.:"
  echo "  sudo apt-get install nsight-systems-2024.4.1"
  echo "  sudo apt-get install nsight-compute-2024.4.1"
  echo "Use:  apt list -a nsight-systems  and  apt list -a nsight-compute  to see what's available."
else
  echo "==> Installed base packages successfully."
fi

echo "==> Verifying tools ..."
if command -v nsys >/dev/null 2>&1; then
  echo -n "nsys version: "; nsys --version || true
else
  echo "nsys not found in PATH."
fi

if command -v ncu >/dev/null 2>&1; then
  echo -n "ncu version: "; ncu --version || true
else
  echo "ncu not found in PATH."
fi

echo "==> Done."
