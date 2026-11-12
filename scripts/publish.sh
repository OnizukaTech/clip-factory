#!/usr/bin/env bash
# publish.sh <branch> <source dir> <target subdir>
# Copies files onto a results branch (job/<id>) so they can be fetched with plain git.
set -euo pipefail
BRANCH="$1"; SRC="$(cd "$2" && pwd)"; SUB="$3"
D="$(mktemp -d)"
cd "$D"
git init -q
git config user.name "clip-factory"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git remote add origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if git fetch -q --depth 1 origin "$BRANCH" 2>/dev/null; then
  git checkout -q -B "$BRANCH" FETCH_HEAD
else
  git checkout -q --orphan "$BRANCH"
fi
mkdir -p "$SUB"
cp -r "$SRC"/. "$SUB"/
git add -A
git commit -qm "$SUB for $BRANCH"
git push -q origin "HEAD:refs/heads/$BRANCH"
echo "published $SUB to $BRANCH"
