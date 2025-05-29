# STEM Web Dataset Preprocessing

This directory contains scripts for preprocessing the STEM web dataset and running the verifier judge.

## Running the Verifier Judge

To run the verifier judge service:

1. Submit the job using sbatch:
   ```bash
   sbatch run_verifier_judge.sh
   ```

2. The script will:
   - Detect the node's IP address automatically
   - Launch a vLLM server running the TIGER-Lab/general-verifier model

3. The service will be available at `http://<node-ip>:8000`

4. User need to set a environment variable using following format: *export STEM_LLM_JUDGE_URL="http://{node_ip}:8000"*