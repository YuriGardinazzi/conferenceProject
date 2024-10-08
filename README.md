# Persistence Topological feature in Large Language Models
## Paper: ***arxiv link***
## With this code you can reproduce the results of the paper and it is subdivided in five main parts
-   Extraction of the hidden representations 
-   ZigZag algorithm execution over the representations
-   Pruning with the results from ZigZag
-   Benchmark with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
-   Extraction of the blocks to cut with the [Angular distance](https://arxiv.org/abs/2403.17887v1) and [Bi-Score](https://arxiv.org/abs/2403.03853) with a modified version of [short-transformers](https://github.com/melisa/short-transformers) (to make it run with Pyhia 6.9B)

## Environment setup

To install the library for FastZigZag refer to their [paper](https://arxiv.org/abs/2204.11080) and their [github folder](https://github.com/TDA-Jyamiti/fzz).

To install the rest of the environment ```conda env create -f environment.yml```