# Performance Record

## Conformer Result Bidecoder (large)
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Pipeline info: run_conformer_bidecoder_large.sh, include data preprocessing, training and testing
* Training info: train_conformer_bidecoder_large.yaml, kernel size 31, lr 0.002, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.3, average_num 10


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 5.85       | 4.08       | **4.55**   |
| ctc prefix beam search           | 5.77+      | 3.90       | 4.68       |
| attention decoder                | 5.96       | 4.09       | 4.96       |
| attention rescoring              | **5.61**+  | **3.78**   | 4.65       |

note that "+" means we removed two <0.1s wav files in test1 before decoding.



## Conformer Result


* 50 epochs
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Pipeline info: run.sh, include data preprocessing, training and testing
* Training info: train_conformer.yaml, kernel size 15, lr 0.004, batch size 12, 8 gpu, acc_grad 1, 50 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 10


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 7.94       | 5.29       | 6.10       |
| ctc prefix beam search           | 7.83+      | 5.28       | 6.08       |
| attention decoder                | 7.83       | 5.63       | 6.37       |
| attention rescoring              | **7.28**+  | **4.81**   | **5.44**   |

note that "+" means we removed two <0.1s wav files in test1 before decoding.


* 120 epochs
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Pipeline info: run.sh, include data preprocessing, training and testing
* Training info: train_conformer.yaml, kernel size 15, lr 0.004, batch size 12, 8 gpu, acc_grad 1, 120 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 10


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 7.66       | 5.15       | 5.83       |
| ctc prefix beam search           | 7.56+      | 5.06       | 5.84       |
| attention decoder                | 7.49       | 5.30       | 6.02       |
| attention rescoring              | **6.97**+  | **4.65**   | **5.29**   |

note that "+" means we removed two <0.1s wav files in test1 before decoding.




## Conformer U2++ Result

* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Pipeline info: run_u2pp_conformer.sh, include data preprocessing, training and testing
* Training info: train_u2++_conformer.yaml, kernel size 15, lr 0.001, batch size 12, 8 gpu, acc_grad 1, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.3, average_num 10


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 6.63       | 4.93       | 5.04       |
| ctc prefix beam search           | 6.59+      | 4.87       | 5.01       |
| attention decoder                | 6.41       | 4.48       | 4.93       |
| attention rescoring              | **6.20**+  | **4.39**   | **4.56**   |

note that "+" means we removed two <0.1s wav files in test1 before decoding.



## Conformer U2 Result


* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Pipeline info: run_unified_conformer.sh, include data preprocessing, training and testing
* Training info: train_unified_conformer.yaml, kernel size 15, lr 0.001, batch size 12, 8 gpu, acc_grad 1, 120 epochs, dither 0.0
* Decoding info: ctc_weight 0.3, average_num 10


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 6.80       | 5.11       | 5.12       |
| ctc prefix beam search           | 6.76+      | 5.03       | 5.11       |
| attention decoder                | 6.39       | 4.61       | 5.25       |
| attention rescoring              | **6.28**+  | **4.43**   | **4.70**   |

note that "+" means we removed two <0.1s wav files in test1 before decoding.

