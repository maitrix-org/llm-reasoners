# COLD-Attack

COLD-Attack adapts the [Energy-based Constrained Decoding with Langevin Dynamics (COLD)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3e25d1aff47964c8409fd5c8dc0438d7-Abstract-Conference.html), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. 

![plot](./assets/COLD_attack_diagram.png)

The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios including:
1. Fluent suffix attacks (standard attack setting which append the adversarial prompt to the original malicious user query).
2. Paraphrase attack with and without sentiment steering (revising a user query adversarially with minimal paraphrasing).
3. Attack with left-right-coherence (inserting stealthy attacks in context with left-right-coherence).

### How to Run

To run COLD-Attack on Advbench dataset, simply using:
1. Fluent suffix attack:
```
python examples/COLD-Attack/inference.py suffix
```

2. Paraphrase attack:
```
python examples/COLD-Attack/inference.py paraphrase
```

3. Left-right-coherence control:
```
python examples/COLD-Attack/inference.py control
```

## Reference
```
@article{guo2024cold,
  title={COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability},
  author={Guo, Xingang and Yu, Fangxu and Zhang, Huan and Qin, Lianhui and Hu, Bin},
  journal={arXiv preprint arXiv:2402.08679},
  year={2024}
}
```
```
@article{qin2022cold,
  title={Cold decoding: Energy-based constrained text generation with langevin dynamics},
  author={Qin, Lianhui and Welleck, Sean and Khashabi, Daniel and Choi, Yejin},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9538--9551},
  year={2022}
}
```
