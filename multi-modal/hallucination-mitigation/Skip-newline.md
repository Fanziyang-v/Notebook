# Skip \n

Skip \n 是一个非常简单且有效的幻觉缓解方法，基于经验发现在**段落符**（两个换行符）之后，多模态大模型产生幻觉的概率显著提高。因此，在换行符前的内容幻觉率**较低**，因此可以通过**避免生成换行符**来缓解幻觉。具体而言，在输入中附加 “in one paragraph” 的限制，提示多模态大模型生成的响应限制在一个段落内；对生成 token 的 logits 进行调整，对于换行符的 logit 设置为**负无穷**，导致 softmax 后换行符的生成概率为 0。通过输入和输出层面的调整，可以有效地避免大模型输出换行符，从而在一定程度上缓解幻觉现象。

![skip newline](./assets/skip-newline.png)



## Reference

1. [Skip \n: A Simple Method to Reduce Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2402.01345) (Feb. 2, 2024, **ICLR 2024 Workshop**)[![arxiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.01345)[![github](https://img.shields.io/github/stars/hanmenghan/Skip-n)](https://github.com/hanmenghan/Skip-n)![alias](https://img.shields.io/badge/SKIP\n-black)