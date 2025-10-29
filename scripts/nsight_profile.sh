
#!/usr/bin/env bash
set -euo pipefail
APP="${1:-./build/deepvision_rtx}"
nsys profile -o nsys_report "$APP"
echo "Generated nsys_report.qdrep"
