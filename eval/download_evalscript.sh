#!/bin/bash

cd eval
echo "installing evaluation dependencies"
echo "downloading e2e-metrics..."
git clone https://github.com/tuetschek/e2e-metrics e2e
pip install -r e2e/requirements.txt

echo "script complete!"
