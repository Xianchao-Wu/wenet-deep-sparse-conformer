贡献者：吴先超，英伟达

recipe: https://github.com/wenet-e2e/wenet/tree/main/examples/csj/s0

关于数据处理的详细描述，请参看：https://zhuanlan.zhihu.com/p/445032888

## 数据集简述：
* CSJ (Corpus of Spontaneous Japanese, https://ccd.ninjal.ac.jp/csj/en/)
* 500 小时，来自于日语学术会议的录音。
* 它是一个收集大量日语自发声音并添加大量研究信息的口语研究数据库，无论从质量还是数量上都是世界上最高水平的日语口语数据库之一。
* 该语料库可以用于语音信息处理、自然语言处理、日语语言学、语言学、语音学、心理学、社会学、日语教育、词典编纂等广泛领域。

## 数据前处理：
* CSJ是XML格式来组织的文本，因为每个wav都是10分钟以上，需要根据XML文件里面的句子信息，分别对XML和WAV进行按照句子切割 (分别使用了csj_tools文件夹下的wn.0.parse.py和wn.1.split_wav.py实现)。
* wav中有40多个文件是双声道的（list_files/2ch.id.list里面），可以统一转换为单声道。
* 按照CSJ的建议，从500小时的语音中，分别找出10个文件，作为test1, test2, test3。这三个测试集合使用的文件id在list_files/test.set.1/2/3.list里面。
* 按照WeNet的要求，为训练集，三个测试集构造text和wav.scp文件。这里的发展集设定为三个测试集的合体。
* 当有时长<0.1s的语音文件的时候，会导致抽取feature的时候，window size小于语音长度的尴尬问题。所以，可以过滤一下，过滤的脚本用的是csj_tools/wn.3.mincut.py
* 因为按照句子划分，最长的在16秒左右，所以这里没有对长度上限进行过滤。
* 词典方面，使用的是sentencepiece，设定了词表大小为4096（基本上是3000多单字加1000多复合词的思路），以及bpe来切割。
* 然后是data.list，这个是WeNet所需要的规范化输入。鉴于CSJ中有几个ID的text和wav不是1：1对齐的，这里简单修改了一下WeNet中的make_raw_list.py，去除了1：1限制，更新后的脚本是csj_tools/wn.4.make_raw_list.py
* 模型的训练，测试，都是沿用了WeNet的标配。

## 结果如下面四个表所示：

## Conformer Result Bidecoder (large)


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 5.85       | 4.08       | 4.55       |
| ctc prefix beam search           | 5.77+      | 3.90       | 4.68       |
| attention decoder                | 5.96       | 4.09       | 4.96       |
| attention rescoring              | 5.61+      | 3.78       | 4.65       |

+号表示在测试的时候，删除了两个长度<0.1s的语音文件。



## Conformer Result


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 7.94       | 5.29       | 6.10       |
| ctc prefix beam search           | 7.83+      | 5.28       | 6.08       |
| attention decoder                | 7.83       | 5.63       | 6.37       |
| attention rescoring              | 7.28+      | 4.81       | 5.44       |

+号表示在测试的时候，删除了两个长度<0.1s的语音文件。


## Conformer U2++ Result


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 6.63       | 4.93       | 5.04       |
| ctc prefix beam search           | 6.59+      | 4.87       | 5.01       |
| attention decoder                | 6.41       | 4.48       | 4.93       |
| attention rescoring              | 6.20+      | 4.39       | 4.56       |

+号表示在测试的时候，删除了两个长度<0.1s的语音文件。



## Conformer U2 (unified conformer) Result



| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 6.80       | 5.11       | 5.12       |
| ctc prefix beam search           | 6.76+      | 5.03       | 5.11       |
| attention decoder                | 6.39       | 4.61       | 5.25       |
| attention rescoring              | 6.28+      | 4.43       | 4.70       |

+号表示在测试的时候，删除了两个长度<0.1s的语音文件。

