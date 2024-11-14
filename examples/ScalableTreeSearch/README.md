


- Need to use [dart-math](https://github.com/hkust-nlp/dart-math) to evaluate the math expressions.
    - `git clone https://github.com/hkust-nlp/dart-math.git`
    - We don't want to install the package (because the dependencies will break the current environment)
    - So we just extract their code so that we can directly import dart_math
    - `cp dart-math/dart_math .`

- `math_500_test.csv` is from "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
However, this file includes the full solution and it's hard to parse the final answer from it. We therefore process our `my_math_500_test.csv` file which only includes the final answers.

- `llama3-math.ipynb` uses SGLang to test Llama3 on MATH (1. standard decoding 2. breaking into steps 3. a prototype of beam search)

- `try_reward_model.ipynb` shows an example of calling an [PRM](https://hanningzhang.github.io/math-prm/). Need to serve it with SGLang and integrate it into the search.