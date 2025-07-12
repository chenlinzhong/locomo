
# locomo

Test the performance of the locomo dataset on different models

# how to use ?

>  make benchmark MODEL=gpt-4.1

the result will save to  **\${output_dir}/\${model}_score.txt**

```
Mean Scores Per Category:
          bleu_score  f1_score  llm_score  count
category                                        
1             0.4178    0.5133     0.6135    282
2             0.3572    0.4100     0.6636    321
3             0.1761    0.2322     0.5417     96
4             0.5855    0.7055     0.8609    841

Overall Mean Scores:
bleu_score    0.4817
f1_score      0.5792
llm_score     0.7545
```

# Visualization

You can paste the output into the output box of this link, and the results will be visualized automatically

[https://yuanbao.tencent.com/sc/dE3IdAJTaHISgB7](https://yuanbao.tencent.com/sc/dE3IdAJTaHISgB7)
