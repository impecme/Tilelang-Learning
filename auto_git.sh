#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"

message="${*:-auto: update $(date '+%Y-%m-%d %H:%M:%S')}"

git add -A
git commit -m "$message"
git push
