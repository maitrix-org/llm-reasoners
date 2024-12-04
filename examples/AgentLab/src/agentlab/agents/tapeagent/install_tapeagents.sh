#!/bin/bash

if [ ! -d "$(dirname "$0")/TapeAgents" ]; then
    # Clone the repository to this directory
    git clone https://github.com/ServiceNow/TapeAgents.git "$(dirname "$0")/TapeAgents"
    # Install the package in editable mode
    pip install -e "$(dirname "$0")/TapeAgents"
else
    echo "TapeAgents directory already exists. Skipping installation."
fi
