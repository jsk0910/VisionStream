#!/bin/bash
echo "==========================================="
echo " Starting VisionStream Streamlit Dashboard"
echo "==========================================="

# Ensure environment is active
source /opt/venv/bin/activate

# Move to the tools directory
cd /home/vml/projects/VisionStream/tools/ui

# Launch Streamlit on port 8501
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
