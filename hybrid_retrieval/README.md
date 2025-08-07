# hybrid_1.py
有时候一个问题的答案无法直接靠词频或语义检索得到，而是处于检索到的句子的附近，即上下文处。
将文档分为子块和父块两个级别，先对子块进行检索，再将其所属的父块扩展进来，尝试解决上述的问题。

### 检索流程
1.文档分块：将原文档先划分为较大的父块，再把每个父块分为较小的子块。在不超出token阈值的前提下，保证每个句子的完整性。每个子块记录了其所属父块的id，便于后续的扩展。
2.混合检索：使用BM25和向量检索器对子块进行检索，得到的结果使用RRF进行排序融合得到子块检索结果。然后获取这些子块的父块，重排序后作为检索的最终结果。

### 组件实现
- 分块器：python实现
- 向量/BM25数据库：Chroma
- BM25检索：NLTK分词，Chroma提供词频数据，python计算
- 向量检索：langchain的HuggingFaceEmbeddings加载嵌入器，Chroma负责存储向量和实现cosine检索
- 重排器：transformer库实现

### 快速开始

安装第三方库

```bash
pip install torch numpy nltk tqdm scikit-learn chromadb langchain-huggingface langchain-community transformers sentence-transformers
```

