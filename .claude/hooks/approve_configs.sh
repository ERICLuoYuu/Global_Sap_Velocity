#!/bin/bash
# approve_configs.sh - Create guard file for stop hook approval
#
# Usage: approve_configs.sh <ppid> <file1> [file2] ...
#
# Creates a JSON guard file with content hashes for each approved file.
# The stop hook checks this file to avoid re-prompting for the same content.
#
# Guard file format:
# {
#   "approved_at": "2026-01-04T20:00:00Z",
#   "files": {
#     ".yamllint": "sha256:abc123...",
#     ".flake8": "sha256:def456..."
#   }
# }

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: approve_configs.sh <ppid> <file1> [file2] ..." >&2
  exit 1
fi

ppid="$1"
shift

guard_file="/tmp/stop_hook_approved_${ppid}.json"

# Build JSON with file hashes
json='{"approved_at":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'","files":{'
first=true

for file in "$@"; do
  [[ ! -f "${file}" ]] && continue
  hash=$(sha256sum "${file}" | cut -d' ' -f1)
  if [[ "${first}" == "true" ]]; then
    json+="\"${file}\":\"sha256:${hash}\""
    first=false
  else
    json+=",\"${file}\":\"sha256:${hash}\""
  fi
done

json+='}}'

echo "${json}" > "${guard_file}"

echo "Guard file created: ${guard_file}"
