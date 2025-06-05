# Instructions on Using Guru Reward Functions in VeRL

This directory provides an example of launching [DAPO](https://arxiv.org/abs/2503.14476) training with Guru reward functions, using the `reward_fn_import.sh` script. The script demonstrates how to integrate a modular reward library (located in `llm-reasoners/reasoners/reward/`) with the VeRL framework for large language models (LLMs).

---

## 1. Overview

- **Goal:**  
  Run DAPO training for LLMs with a custom reward function.  
- **Approach:**  
  - Import a reward function from the local reward library at runtime.  
  - Pass its path and function name to VeRL via environment variables.  
  - Launch DAPO training using the `reward_fn_import.sh` script.

---

## 2. Environment Setup

### 2.1 Create a Conda Environment

We recommend using Conda to manage dependencies:

```bash
conda create -n verl python=3.10 -y
conda activate verl
````

### 2.2 Install VeRL Dependencies

After activating the `verl` environment, install VeRL and its core dependencies:

1. **Run the install script** (choose either Megatron or FSDP backend):

   * **With Megatron (default):**

     ```bash
     bash scripts/install_vllm_sglang_mcore.sh
     ```

   * **With FSDP (no Megatron):**

     ```bash
     USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
     ```

   > If you encounter errors, inspect the script and manually install any missing packages.

2. **Clone and install VeRL from source:**

   ```bash
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install --no-deps -e .
   ```

3. **Install additional Python packages:**

   ```bash
   pip install vllm==0.8.5
   pip install tensordict==0.6.2
   pip install datasets transformers numpy polars pandas rich tqdm matplotlib sympy pylatexenc requests
   ```

---

## 3. How the Reward Library Is Used

VeRL accepts a custom reward function by specifying two arguments:

* `custom_reward_function.path`
  Path to the Python file containing the reward logic.

  ```bash
  custom_reward_function.path=llm-reasoners/reasoners/reward/__init__.py
  ```

* `custom_reward_function.name`
  Name of the function to invoke as the reward entry point.

  ```bash
  custom_reward_function.name=_default_compute_score
  ```

You can modify these values in `reward_fn_import.sh` to point to any reward function you prefer. For example, to use a function defined in `naive_dapo.py`:

```bash
custom_reward_function.path=llm-reasoners/reasoners/reward/naive_dapo.py
custom_reward_function.name=compute_score
```

> **Tip:** Ensure your custom function’s signature matches VeRL’s expected interface (see `_default_compute_score` for reference).

---

## 4. How to Run

1. **Prepare Your Data**

   * Download the training dataset (Parquet format) from [Huggingface](https://huggingface.co/datasets/LLM360/guru_RL_verl).
   * Place all `.parquet` files under `data/parquet/` (or update `DATA_DIR` in the script).

2. **Set the Base Model**

   * By default, `BASE_MODEL` is set to `Qwen/Qwen2.5-7B-Instruct`.
   * Open `reward_fn_import.sh` and change `BASE_MODEL` if you want to use a different model.

3. **Configure Additional Parameters**

   * Inside `reward_fn_import.sh`, adjust any training hyperparameters (batch size, maximum prompt/response length, etc.) as needed.
   * Specify your custom reward function via `custom_reward_function.path` and `custom_reward_function.name`.

4. **Launch Training**
   Run the script:

   ```bash
   bash reward_fn_import.sh
   ```

   The script will:

   * Set up environment variables
   * Collect all Parquet files in `DATA_DIR`
   * Import the specified reward function
   * Launch DAPO training with VeRL

---

## 5. Example Command

Simply run:

```bash
bash reward_fn_import.sh
```

This will start DAPO training using:

* `Qwen/Qwen2.5-7B-Instruct` as the base model
* The reward function defined in `__init__.py` (default entry point `_default_compute_score`)
* All training data in `data/parquet/`

---

## 6. Customization Tips

* **Switch Reward Functions**
  Edit these lines in `reward_fn_import.sh`:

  ```bash
  custom_reward_function.path=llm-reasoners/reasoners/reward/your_module.py
  custom_reward_function.name=your_compute_function
  ```

* **Change Base Model**

  ```bash
  BASE_MODEL="your_org/your_model-name"
  ```

* **Adjust Training Parameters**
  Modify variables like `BATCH_SIZE`, `MAX_PROMPT_LENGTH`, or `LEARNING_RATE` in the script.

* **Use a Different Data Directory**
  Set:

  ```bash
  DATA_DIR="path/to/your/parquets"
  ```

