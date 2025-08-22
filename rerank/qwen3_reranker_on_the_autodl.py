from huggingface_hub import snapshot_download
import logging
from typing import Dict, Optional, List
import os

import json
import logging
import os
import queue
import sys

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data._utils.worker import ManagerWatchdog

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    is_torch_npu_available,
)

class Qwen3Reranker:
    LOCAL_BASE = "/root/autodl-tmp/"
    
    def __init__(
        self,
        model_name: str,
        local_dir: str,
        max_length: int = 4096,
        instruction=None,
        attn_type="causal",
        use_cuda: bool = None,
    ) -> None:
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        local_dir = self.LOCAL_BASE+"/"+local_dir
        
        if not os.path.isdir(local_dir):
            print(f"没有模型，开始下载{model_name}")
            self.download_model(model_name, local_dir)
    
            
        
        print(f"正在加载Qwen3 Reranker模型...")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"使用CUDA: {use_cuda}")
        
        n_gpu = torch.cuda.device_count()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True, padding_side="left"
        )
        
        dtype = torch.float16 if use_cuda else torch.float32
        
        self.lm = AutoModelForCausalLM.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            # attn_implementation="flash_attention_2",
        )
        
        if use_cuda:
            self.lm = self.lm.cuda()
        else:
            self.lm = self.lm.cpu()
            
        self.lm = self.lm.eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n</think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )
        self.instruction = instruction
        if self.instruction is None:
            self.instruction = "Given the user query, retrieval the relevant passages"

    def download_model(self, model_name, local_dir):
    
        # 镜像网站URL
        mirror_url = "https://hf-mirror.com"
        
        # 创建保存目录（如果不存在）
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            print(f"开始从{mirror_url}下载模型 {model_name}...")
            # 配置镜像源并下载模型
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
                resume_download=True,         # 支持断点续传
                max_workers=4,                # 下载线程数
                endpoint=mirror_url
            )
            print(f"模型已成功下载到 {os.path.abspath(local_dir)}")
        except Exception as e:
            print(f"下载过程中出现错误: {str(e)}")
        
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = self.instruction
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs):
        out = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(out["input_ids"]):
            out["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        out = self.tokenizer.pad(
            out, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in out:
            out[key] = out[key].to(self.lm.device)
        return out

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.lm(** inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def compute_scores(self, pairs, instruction=None, **kwargs):
        # 保存原始的查询和文档对
        original_pairs = pairs.copy()
        
        # 处理输入并计算分数
        formatted_pairs = [
            self.format_instruction(instruction, query, doc) for query, doc in pairs
        ]
        inputs = self.process_inputs(formatted_pairs)
        scores = self.compute_logits(inputs)
        
        # 构建包含分数和文本的结果列表
        results = []
        for i in range(len(scores)):
            query, doc = original_pairs[i]
            results.append({
                "score": scores[i],
                "query": query,
                "document": doc
            })
        
        return results
    
    def rerank(self, query: str, passages: List[str], instruction=None) -> List[Dict]:
        """
        计算单个查询与多个段落的相似度，并按分数降序返回排序结果
        
        Args:
            query: 单个查询语句
            passages: 多个文档段落的列表
            instruction: 可选的指令文本
            
        Returns:
            按相似度分数降序排列的结果列表，每个元素包含分数、查询和文档
        """
        # 创建查询与每个段落的配对
        pairs = [(query, passage) for passage in passages]
        
        # 计算分数
        results = self.compute_scores(pairs, instruction)
        
        # 按分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results


# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # 模型配置
    model_name = "Qwen/qwen3-reranker-0.6B"
    local_dir = "Qwen3-Reranker-0.6B"
    
    # 初始化重排序模型
    model = Qwen3Reranker(
        model_name=model_name,
        local_dir=local_dir,
        instruction="Given the user query, retrieval the relevant passages",
        max_length=2048,
    )
    
    # 测试数据
    query = "什么是人工智能"
    passages = [
        "人工智能是计算机科学的一个分支，研究如何使机器模拟人类智能",
        "机器学习是人工智能的一个重要子领域，专注于开发能从数据中学习的算法",
        "量子计算是一种基于量子力学原理的计算方式，具有潜在的超强计算能力",
        "人工智能应用广泛，包括图像识别、自然语言处理、自动驾驶等领域",
        "气候变化是当今世界面临的重大挑战之一，影响着全球生态系统"
    ]
    
    # 进行重排序
    reranked_results = model.rerank(query, passages)
    
    # 打印排序结果
    print(f"查询: {query}\n")
    print("重排序结果 (按相似度降序):")
    for i, result in enumerate(reranked_results, 1):
        print(f"排名 {i}:")
        print(f"  分数: {result['score']:.6f}")
        print(f"  文档: {result['document']}\n")
