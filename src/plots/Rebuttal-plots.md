# Rebuttal Plots
New benchmarks at the end.
## Persistence 
![Llama2 7b](Llama2_7b_pers.png)
![Mistral](mistral_pers.png)
![Pythia](pythia_pers.png)

Persistence computed on the SST Dataset with 16 splits as in Fig. 3 for the models Llama 2 7B, Mistral and Pythia

## Birth Relative Frequency

![Llama2 7b](llama2_7b_16_split.png)
![Mistral](mistral_16_split.png)
![Pythia](pythia_16_split.png)

Birth Relative Frequency computed on the SST Dataset with 16 splits as in Fig. 3 for the models Llama 2 7B, Mistral and Pythia


## Inter Layer Persistance
![ILP Llama2 7b](ILP_llama2_7b_16_split.png)
![ILP Mistral](ILP_mistral_16_split.png)
![ILP Pythia](ILP_pythia_16_split.png)

Inter Layer Persistance computed on the SST Dataset with 16 splits as in Fig. 3 for the models Llama 2 7B, Mistral, Pythia

# Benchmarks

| Model   | CMMLU |            |            | Commonsense-QA |           |             | WSC   |           |             |
|---------|-------|-----------|-------------|----------------|-----------|-------------|-------|-----------|-------------|
|         | full  | this work | other works | full           | this work | other works | full  | this work | other works |
| llama 2 | 32.17 | 28.51     | 30.00       | 55.94          | 38.40     | 52.83       | 88.63 | 84.60     | 75.80       |
| llama 3 | 50.96 | 34.02     | 34.02       | 73.45          | 65.93     | 65.93       | 85.70 | 80.94     | 80.94       |
| Mistral | 44.47 | 38.84     | 29.68       | 69.78          | 62.33     | 30.62       | 87.18 | 69.60     | 72.15       |
| Pythia  | -     | -         | -           | 19.74          | 19.97     | 20.64       | 81.67 | 60.06     | 71.43       |