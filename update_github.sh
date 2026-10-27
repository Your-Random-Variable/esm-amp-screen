#!/usr/bin/env bash
set -e

# Change directory to the repository root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "==========================================="
echo "   ESM-AMP-Screen GitHub Update Script"
echo "==========================================="

# Ensure SSH over port 443 is configured (avoids firewall timeout on port 22)
git remote set-url origin ssh://git@ssh.github.com:443/Your-Random-Variable/esm-amp-screen.git

# Check if there are any changes
if [ -z "$(git status --porcelain)" ]; then
    echo "✔ Working tree is clean. Nothing to commit."
    exit 0
fi

# Show changed files
echo "Changed files:"
git status -s

# Prompt or use supplied commit message
COMMIT_MSG="$1"
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="update: routine code maintenance and improvements"
fi

# Stage all files
git add -A

# Commit
git commit -m "$COMMIT_MSG"

# Push to main
echo "Pushing commits to origin/main..."
git push origin main

# Optional tag argument: ./update_github.sh "commit msg" "v1.1.3"
TAG_NAME="$2"
if [ -n "$TAG_NAME" ]; then
    echo "Creating and pushing tag: $TAG_NAME"
    git tag -a "$TAG_NAME" -m "Release $TAG_NAME"
    git push origin "$TAG_NAME"
fi

echo "==========================================="
echo "✔ Successfully updated GitHub!"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "  Branch: main"
echo "==========================================="
