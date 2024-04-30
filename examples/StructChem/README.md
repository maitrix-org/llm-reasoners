# Examples

## StructChem

**Setup**

Please download [SciBench](https://github.com/mandyyyyii/scibench) dataset and put the test set directly under this folder like this:

```bash
./examples/StructChem/chemmc.json
```

**How to Run**

To run StructChem on the datasets, simply using:

```bash
OPENAI_API_KEY="your_api_key" python examples/StructChem/inference.py --data_path "examples/StructChem/chemmc.json"
```

The "iterative review and refinement" is implemented as "tree search". Defaultly, we use beam search here,

```python
search_algo_params |= {
            'beam_size': beam_size, 
            'max_depth': max_depth,
            }
```

where "max_depth" controls the number of iterations of this process.

**Evaluation**

The program will return the final accuracy with integration of evaluation snippet as "quasi exact match".