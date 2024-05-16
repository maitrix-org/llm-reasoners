## PAL + Guided Beam Search

> Note: If you want to use `code-davinci-002`, you need to apply for the [research access](https://openai.com/form/researcher-access-program) to `Codex` (`code-davinci-002`) to run this approach

#### Beam Search (Stochastic Sampling) 
For `code-davinci-002`
```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/Self-Eval/gsm8k/inference.py --n_actions 16 --temperature 1.0 --reward_alpha 0.5 \
    --beam_size 5 --sampling_strategy stochastic --replace False \
    --reject_sample True --reject_min_reward 0.6 --unbiased True \
    --beam_search_temperature 0.5 --beam_search_temperature_decay 1.0 \
    --majority_voting_n 4 
```

#### Beam Search (Top-k Sampling) 
For `code-davinci-002`
```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/Self-Eval/gsm8k/inference.py --n_actions 16 --temperature 1 --reward_alpha 0.5 \
    --beam_size 5 --sampling_strategy argmax 

```
