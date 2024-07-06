#!/bin/bash

REPO_URL="https://github.com/qlinhta/llm-psl.git"
TOKEN="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"
CLONE_URL="https://$TOKEN@${REPO_URL:8}"

git clone $CLONE_URL
