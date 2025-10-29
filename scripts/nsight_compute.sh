
#!/usr/bin/env bash
set -euo pipefail
APP="${1:-./build/deepvision_rtx}"
ncu --set full --target-processes all "$APP"
