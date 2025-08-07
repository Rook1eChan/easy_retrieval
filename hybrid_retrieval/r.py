import os
import math
import uuid
import json
import pickle
import shutil
import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from chromadb import PersistentClient, Collection
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 清理旧数据
def clear_old_data(db_paths):
    """清理文件夹及文件"""
    for name, path in db_paths.items():
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"已清理 {name}: {path}")

# 句子感知分割器（不截断句子）
def split_into_chunks(text, chunk_size):
    """
    将文本分割为指定大小的块，不截断句子
    按换行符分割，确保每个块都是完整句子的集合
    """
    # 按换行符分割成句子/段落
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 检查当前块加上新句子是否超过指定大小
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:  # +1 是为了换行符
            current_chunk += sentence + "\n"
        else:
            if current_chunk:  # 添加当前块
                chunks.append(current_chunk.strip())
            # 处理超长句子（虽然题目说不要截断，但极端情况做保护）
            if len(sentence) > chunk_size:
                # 超长句子强制分割（尽量避免）
                sub_chunks = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size)]
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] + "\n"
            else:
                current_chunk = sentence + "\n"
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# 批量添加文档到Chroma
def batch_add_to_chroma(collection, docs, embeddings, batch_size=500):
    """
    批量将文档添加到Chroma
    docs格式: [{"id": "xxx", "text": "xxx", "metadata": {...}}]
    """
    if not docs:
        return 0
    
    total = len(docs)
    added = 0
    
    # 按批次处理
    for i in range(0, total, batch_size):
        batch = docs[i:i+batch_size]
        
        # 提取批次数据
        ids = [doc["id"] for doc in batch]
        texts = [doc["text"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        
        # 计算嵌入
        embeddings_list = embeddings.embed_documents(texts)
        
        # 添加到集合
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        added += len(batch)
        print(f"已添加 {added}/{total} 个文档 (批次 {i//batch_size + 1})")
    
    return added

# 处理单个文档为父子块结构
def process_document(doc, parent_chunk_size=1500, child_chunk_size=300):
    """
    处理单个文档，生成父块和子块
    返回: (parent_chunks, child_chunks)
    """
    # 1. 生成父块
    parent_texts = split_into_chunks(doc.page_content, parent_chunk_size)
    parent_chunks = []
    
    for i, text in enumerate(parent_texts):
        parent_id = f"parent_{uuid.uuid4().hex[:12]}"  # 生成唯一父ID
        parent_chunks.append({
            "id": parent_id,
            "text": text,
            "metadata": {
                "source": doc.metadata.get("source", ""),  # doc的文件名
                "chunk_type": "parent",
                "chunk_index": i
            }
        })
    
    # 2. 为每个父块生成子块
    child_chunks = []
    for parent in parent_chunks:
        child_texts = split_into_chunks(parent["text"], child_chunk_size)
        
        for j, text in enumerate(child_texts):
            child_id = f"child_{uuid.uuid4().hex[:12]}"  # 生成唯一子ID
            child_chunks.append({
                "id": child_id,
                "text": text,
                "metadata": {
                    "parent_id": parent["id"],  # 关键：子块指向父块的链接
                    "source": parent["metadata"]["source"],  # doc的文件名
                    "chunk_type": "child",
                    "parent_index": parent["metadata"]["chunk_index"],
                    "child_index": j
                }
            })
    
    return parent_chunks, child_chunks

def process_document_parallel(docs, max_workers=4):
    """多线程处理文档"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda doc: process_document(doc, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE),
            docs
        ))
    return results

def rerank(query, documents, reranker, top_n=150):
    """
    直接使用HuggingFace模型进行重排
    :param query: 查询文本
    :param documents: 待排序文档列表 ["doc1", "doc2"...]
    :param reranker: 加载的重排模型字典(包含model/tokenizer/device)
    :param top_n: 返回数量
    :return: 重排后的文档列表
    """
    if not documents or not query:
        return documents[:top_n]
    
    model = reranker['model']
    tokenizer = reranker['tokenizer']
    device = reranker['device']
    
    # 准备输入对
    pairs = [[query, doc] for doc in documents]
    
    try:
        # 批量tokenize
        features = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)
        
        # 计算分数
        with torch.no_grad():
            scores = model(**features).logits.squeeze(dim=1)
        
        # 转换为numpy数组并排序
        scores = scores.cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1]  # 降序
        
        # 返回排序后的文档
        return [documents[i] for i in sorted_indices[:top_n]]
    
    except Exception as e:
        print(f"重排失败: {str(e)}")
        return documents[:top_n]  # 失败时返回原始顺序
        
# 检索子块并找到对应的父块
def retrieve_child(child_collection, queries, embeddings, top_k=3):
    """检索相关子块"""
    # 1. 检索相关子块
    query_embeddings = embeddings.embed_documents(queries)
    child_results = child_collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k
    )
    
    # 提取子块信息
    child_docs = []
    for i in range(len(child_results["ids"][0])):
        child_docs.append({
            "id": child_results["ids"][0][i],
            "text": child_results["documents"][0][i],
            "metadata": child_results["metadatas"][0][i],
            "distance": child_results["distances"][0][i]
        })
    
    return child_docs

def process_single_query(
    qa, 
    child_collection, 
    parent_collection, 
    embeddings, 
    reranker, 
    bm25_model=None,
    use_hybrid=True,
    use_rerank=True,
    use_father=True,
    child_top_k=100,
    final_top_k=150
):
    """
    改进版检索流程：
    1. 先检索子块（向量/B混合）
    2. RRF融合排序
    3. 扩展父块
    4. 最终重排
    
    参数：
    - bm25_model: BM25索引对象
    - use_hybrid: 是否使用混合检索
    - use_rerank: 是否使用重排
    - child_top_k: 子块检索数量
    - final_top_k: 最终返回数量
    """
    query = qa["question"]
    try:
        # ===== 第一阶段：子块检索 =====
        if use_hybrid:
            # 混合检索流程
            # 1. 向量检索子块
            vector_child = retrieve_child(
                child_collection, [query], embeddings, top_k=child_top_k
            )
            
            # 2. BM25检索子块
            bm25_child = bm25_search(bm25_model, query, child_collection, top_k=child_top_k)
            
            # 3. RRF融合
            rrf_scores = {}
            k = 60  # RRF常数
            
            # 向量结果评分
            for rank, doc in enumerate(vector_child, 1):
                text = doc["text"]
                rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k + rank)
            
            # BM25结果评分
            for rank, doc in enumerate(bm25_child, 1):
                text = doc["text"]
                rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k + rank)
            
            # 按RRF分数排序
            sorted_child = sorted(
                vector_child + bm25_child,
                key=lambda x: rrf_scores.get(x["text"], 0),
                reverse=True
            )[:child_top_k]
        else:
            # 纯向量检索
            sorted_child = retrieve_child(
                child_collection, [query], embeddings, top_k=child_top_k
            )
        
        # ===== 第二阶段：扩展父块 =====
        # 获取子块对应的父块ID（去重）
        parent_ids = list({
            doc["metadata"]["parent_id"] 
            for doc in sorted_child 
            if "parent_id" in doc["metadata"]
        })
        
        # 检索父块
        parent_docs = []
        if use_father:  # 扩展父块
            if parent_ids:
                parent_results = parent_collection.get(ids=parent_ids)
                parent_docs = [{
                    "id": parent_results["ids"][i],
                    "text": parent_results["documents"][i],
                    "metadata": parent_results["metadatas"][i]
                } for i in range(len(parent_results["ids"]))]
        
        # ===== 第三阶段：结果合并与处理 =====
        # 合并结果文本（保留原始顺序）
        all_texts = []
        for doc in sorted_child + parent_docs:
            if isinstance(doc, dict) and "text" in doc:
                all_texts.extend(doc["text"].split("\n"))
        
        # 可选重排
        if use_rerank:
            reranked_texts = rerank(query, all_texts, reranker, final_top_k)
        else:
            reranked_texts = all_texts
        
        # 去重并截断
        seen = set()
        qa["retrieved"] = [
            t for t in reranked_texts 
            if not (t in seen or seen.add(t))
        ][:final_top_k]
        
    except Exception as e:
        print(f"处理查询 '{query[:20]}...' 时出错: {str(e)}")
        qa["retrieved"] = []
    return qa

def calculate_metrics(qa, top_k=100, ndcg_k=10):
    """计算单个QA对的评估指标（对比retrieved和evidences）"""
    retrieved = qa.get("retrieved", [])
    gold_evidences = qa.get("evidences", [])  # 黄金段落列表
    
    if not gold_evidences or not retrieved:
        return {
            "recall@100": 0.0,
            "ndcg@10": 0.0,
            "matched_evidences": []
        }
    
    # 计算recall@100：检索结果中有多少黄金段落
    matched = []
    for i, doc in enumerate(retrieved[:top_k]):
        if doc in gold_evidences:
            matched.append({
                "rank": i + 1,
                "text": doc,
                "gold_position": gold_evidences.index(doc) + 1
            })
    
    recall = len(matched) / len(gold_evidences)
    
    # 计算NDCG@10：需要构造相关性向量
    y_true = np.zeros(ndcg_k)
    y_score = np.zeros(ndcg_k)
    
    for i, doc in enumerate(retrieved[:ndcg_k]):
        if doc in gold_evidences:
            # 相关性得分可以根据黄金段落的顺序加权（越靠前的黄金段落权重越高）
            relevance = 1.0 / (gold_evidences.index(doc) + 1)  
            y_true[i] = relevance
            y_score[i] = 1.0  # 预测得分
    
    ndcg = ndcg_score([y_true], [y_score]) if np.any(y_true) else 0.0
    
    return {
        "recall@100": float(recall),
        "ndcg@10": float(ndcg),
        "matched_evidences": matched,
        "gold_evidence_count": len(gold_evidences),
        "retrieved_count": len(retrieved)
    }

def evaluate(result_path, max_workers=8):
    """并发评估（对比evidences）并写回文件"""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 确保数据结构正确
    if isinstance(data, dict) and "details" in data:
        qas = data["details"]
    else:
        qas = data
    
    # 并发计算指标
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        metrics = list(tqdm(
            executor.map(calculate_metrics, qas),
            total=len(qas),
            desc="评估进度"
        ))
    
    # 合并结果
    for qa, m in zip(qas, metrics):
        qa.update(m)
    
    # 计算全局指标
    avg_recall = np.mean([m["recall@100"] for m in metrics])
    avg_ndcg = np.mean([m["ndcg@10"] for m in metrics])
    total_matched = sum(len(m["matched_evidences"]) for m in metrics)
    total_gold = sum(m["gold_evidence_count"] for m in metrics)
    
    summary = {
        "avg_recall@100": avg_recall,
        "avg_ndcg@10": avg_ndcg,
        "total_matched_evidences": total_matched,
        "total_gold_evidences": total_gold,
        "match_ratio": total_matched / total_gold if total_gold > 0 else 0,
        "evaluated_queries": len(qas)
    }
    
    # 写回文件（保留原始结构）
    output = {
        "summary": summary,
        "details": qas
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果：")
    print(f"- Recall@100: {avg_recall:.4f} （匹配到 {total_matched}/{total_gold} 黄金段落）")
    print(f"- NDCG@10: {avg_ndcg:.4f}")
    print(f"结果已保存到 {result_path}")

def load_reranker(model_name=None, device=None):
    """加载本地重排模型和tokenizer"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device
    }

def build_bm25_index(collection, save_path="./bm25_child.json"):
    """构建BM25索引"""
    # 获取所有文档及其ID
    all_docs = collection.get()
    documents = all_docs["documents"]
    doc_ids = all_docs["ids"]  # 保存原始UUID
    
    # 英文分词处理
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    tokenized_docs = []
    for doc in documents:
        tokens = [
            ps.stem(token.lower()) 
            for token in word_tokenize(doc) 
            if token.isalnum() and token.lower() not in stop_words
        ]
        tokenized_docs.append(tokens)
    
    # 计算文档频率(DF)
    df = defaultdict(int)
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] += 1
    
    # 计算逆文档频率(IDF)
    N = len(documents)
    idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in df}
    
    # 计算文档长度
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lengths) / N
    
    # 保存索引（包含原始ID映射）
    index = {
        "doc_ids": doc_ids,  # 关键：保存原始UUID
        "tokenized_docs": tokenized_docs,
        "df": dict(df),
        "idf": idf,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "k1": 1.5,
        "b": 0.75
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"BM25索引已构建，包含 {len(doc_ids)} 个文档")
    return index

def load_bm25_index(save_path="./bm25_child.json"):
    """从本地加载BM25索引"""
    with open(save_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # 兼容性检查
    if len(index["doc_ids"]) != len(index["tokenized_docs"]):
        raise ValueError("文档ID与分词结果数量不匹配")
        
    return index

def bm25_search(index, query, collection, top_k=150):
    """BM25搜索（支持UUID）"""
    # 验证索引
    if "doc_ids" not in index:
        raise ValueError("索引缺少doc_ids字段，请使用新版build_bm25_index重建")
        
    # 分词查询
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    query_terms = [
        ps.stem(token.lower())
        for token in word_tokenize(query)
        if token.isalnum() and token.lower() not in stop_words
    ]
    
    # 准备数据
    tokenized_docs = index["tokenized_docs"]
    idf = index["idf"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avgdl"]
    k1 = index["k1"]
    b = index["b"]
    
    # 计算分数
    scores = []
    for i, doc in enumerate(tokenized_docs):
        doc_len = doc_lengths[i]
        score = 0
        for term in query_terms:
            if term not in idf:
                continue
            tf = doc.count(term)
            numerator = idf[term] * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += numerator / denominator
        scores.append((i, score))
    
    # 按分数排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 获取对应的UUID
    top_ids = [index["doc_ids"][i] for i, _ in scores[:top_k]]
    top_indices = [i for i, _ in scores[:top_k]]
    top_ids = [index["doc_ids"][i] for i in top_indices]
    
    # 批量获取文档
    try:
        results = collection.get(ids=top_ids)
        return [{
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
            "score": scores[i][1]
        } for i in range(len(results["ids"]))]
    except Exception as e:
        print(f"文档获取失败: {str(e)}")
        return []

if __name__ == "__main__":
    # 1.配置参数
    PARENT_CHUNK_SIZE = 1500  # 父块大小
    CHILD_CHUNK_SIZE = 300    # 子块大小
    BATCH_SIZE = 2000          # 批量处理大小
    DB_PATHS = {
        "child": "./child_chroma",
        "parent": "./parent_chroma",
        "bm25": "./bm25_child.json"
    }
    CORPUS_PATH = ["cqa_title.txt"]
    RETRIEVE_TOPK = 100  # 子块检索数量
    FINAL_TOPK = 100  # 最终检索数量
    MAX_WORKERS = min(3, os.cpu_count())  # retrieve工作线程数
    USE_HYBRID = True
    USE_RERANK = True
    USE_FATHER = False
    CLEAN_HISTORY = True
    
    # 2.清理旧数据
    if CLEAN_HISTORY:
        clear_old_data(DB_PATHS)

    # 3.初始化各个组件
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        # model_name="BAAI/bge-small-zh-v1.5",
        model_name="facebook/dragon-plus-context-encoder",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 初始化重排器
    reranker = load_reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 初始化Chroma客户端和集合
    child_client = PersistentClient(path=DB_PATHS["child"])
    parent_client = PersistentClient(path=DB_PATHS["parent"])

    child_collection = child_client.get_or_create_collection(
        name="child_chunks",
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    parent_collection = parent_client.get_or_create_collection(
        name="parent_chunks",
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    
    # 检查是否需要加载文档（如果集合为空）
    if child_collection.count() == 0 or parent_collection.count() == 0:
        print("数据库为空，开始加载并处理文档...")
        
        # 加载所有文档
        docs = []
        for corpus_path in CORPUS_PATH:
            docs.extend(TextLoader(corpus_path).load())
        
        # 依次处理每个文档
        all_parent_chunks = []
        all_child_chunks = []

        # 并发处理所有文档
        all_results = process_document_parallel(docs, max_workers=64)
        all_parent_chunks = [p for res in all_results for p in res[0]]
        all_child_chunks = [c for res in all_results for c in res[1]]
        print(f"共生成 {len(all_parent_chunks)} 个父块和 {len(all_child_chunks)} 个子块")

        # 原：顺序处理每个文档
        # for doc_idx, doc in enumerate(docs):
        #     print(f"处理文档 {doc_idx + 1}/{len(docs)}")
        #     parents, children = process_document(
        #         doc, 
        #         parent_chunk_size=PARENT_CHUNK_SIZE,
        #         child_chunk_size=CHILD_CHUNK_SIZE
        #     )
        #     all_parent_chunks.extend(parents)
        #     all_child_chunks.extend(children)
        #     print(f"生成 {len(parents)} 个父块和 {len(children)} 个子块")
        
        # 批量添加父块到数据库
        print("\n添加父块到数据库...")
        parent_count = batch_add_to_chroma(
            parent_collection, all_parent_chunks, embeddings, BATCH_SIZE
        )
        
        # 批量添加子块到数据库
        print("\n添加子块到数据库...")
        child_count = batch_add_to_chroma(
            child_collection, all_child_chunks, embeddings, BATCH_SIZE
        )
        
        print(f"\n处理完成：共 {parent_count} 个父块，{child_count} 个子块")
    else:
        print("数据库已存在，直接使用现有数据")
        print(f"父块数量: {parent_collection.count()}, 子块数量: {child_collection.count()}")

    # 初始化BM25
    if not os.path.exists(DB_PATHS["bm25"]):
        print("构建BM25索引...")
        # 初始化NLTK组件
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        bm25_model = build_bm25_index(child_collection, DB_PATHS["bm25"])
    else:
        print("加载已有BM25索引...")
        bm25_model = load_bm25_index(DB_PATHS["bm25"])
    
    # 4.进行检索
    with open("new_dev.json", "r", encoding="utf-8") as f:
        qas = json.load(f)

    # 并行处理所有查询
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_func = lambda qa: process_single_query(
            qa=qa,
            child_collection=child_collection,
            parent_collection=parent_collection,
            embeddings=embeddings,
            reranker=reranker,
            bm25_model=bm25_model,
            use_hybrid=USE_HYBRID,
            use_rerank=USE_RERANK,
            use_father=USE_FATHER,
            child_top_k=RETRIEVE_TOPK,
            final_top_k=FINAL_TOPK
        )
        results = list(tqdm(executor.map(process_func, qas), total=len(qas), desc="处理查询"))

    # 5.保存结果
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("检索完成，结果已保存到 result.json")

    # 6.进行评估
    evaluate("result.json", max_workers=MAX_WORKERS)
    print("评估完成，结果已保存到 result.json")
    
              
