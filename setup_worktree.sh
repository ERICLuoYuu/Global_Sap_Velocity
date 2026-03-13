#!/bin/bash
# Setup a git worktree with symlinked data directories
# Usage: ./setup_worktree.sh <branch-name> <worktree-short-name>
# Example: ./setup_worktree.sh feature/aoa-uncertainty aoa
set -euo pipefail

BRANCH="${1:?Usage: $0 <branch> <name>}"
WT_NAME="${2:?Usage: $0 <branch> <name>}"
WT_DIR="/scratch/tmp/yluo2/gsv-wt/${WT_NAME}"
OLD_DATA="/scratch/tmp/yluo2/Global_Sap_Velocity"

git worktree add "${WT_DIR}" "${BRANCH}"

cd "${WT_DIR}"
ln -s "${OLD_DATA}/data" data
ln -s "${OLD_DATA}/outputs" outputs

if [ ! -f path_config.py ]; then
    cp "${OLD_DATA}/path_config.py" . 2>/dev/null || true
fi

echo "Worktree ready at ${WT_DIR}"
echo "  data -> $(readlink data)"
echo "  outputs -> $(readlink outputs)"
