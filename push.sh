#!/bin/bash

REPO_URL="https://github.com/qlinhta/llm-psl.git"
TOKEN="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"
EMAIL="qlinhta@outlook.com"
USERNAME="Quyen Linh TA"
COMMIT_MESSAGE="Update:: root@qlta --root V100-SXM3-32GB --push"

git config --global user.email "$EMAIL"
git config --global user.name "$USERNAME"

git add -A

git commit -m "$COMMIT_MESSAGE"

git push https://$TOKEN@${REPO_URL:8}
