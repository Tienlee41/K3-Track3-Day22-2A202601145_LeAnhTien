$ErrorActionPreference = "Stop"

pref-lab validate data/sample_preferences.jsonl
pref-lab train --config configs/local.yaml
pref-lab evaluate --config configs/local.yaml
Get-Content -Raw outputs/metrics.json
