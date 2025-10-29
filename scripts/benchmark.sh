
#!/usr/bin/env bash
set -euo pipefail
APP="${1:-./build/deepvision_rtx}"
echo "Running: $APP"
"$APP" | tee run.log
