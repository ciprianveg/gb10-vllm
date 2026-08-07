#!/bin/bash
# Create symlink from venv site-packages to dist-packages for mod compatibility.
# v18 image uses /opt/venv; our mods hardcode /usr/local/lib/python3.12/dist-packages.
set -eux

VENV_SITE="/opt/venv/lib/python3.12/site-packages"
DIST_PACKAGES="/usr/local/lib/python3.12/dist-packages"

if [ -d "$VENV_SITE" ] && [ ! -e "$DIST_PACKAGES" ]; then
    mkdir -p "$(dirname "$DIST_PACKAGES")"
    ln -s "$VENV_SITE" "$DIST_PACKAGES"
    echo "✓ Symlinked $DIST_PACKAGES -> $VENV_SITE"
elif [ -d "$VENV_SITE" ] && [ -d "$DIST_PACKAGES" ]; then
    echo "✓ Both paths exist (dist-packages may already be symlinked or real)"
else
    echo "⚠ venv site-packages not found at $VENV_SITE — mods may fail"
fi
