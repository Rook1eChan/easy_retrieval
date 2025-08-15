import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Reranker:
    """
    文档重排器类，用于根据查询与文档的相关性对文档进行排序
    通过计算文档与查询的相关概率（生成'yes'的概率）作为排序依据
    """
    
    def __init__(self, model_path: str = './model', device: str = None):
        """
        初始化重排器
        
        Args:
            model_path: 重排模型路径
            device: 运行设备，默认为自动检测（cuda优先）
        """
        self.model_path = os.path.expanduser(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 4096  # 可根据GPU显存调整
        
        # 模型相关变量初始化
        self.tokenizer = None
        self.model = None
        self.token_true_id = None  # 'yes'的token ID
        self.token_false_id = None  # 'no'的token ID
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载重排模型和分词器"""
        print(f"正在加载模型: {self.model_path} (设备: {self.device})")
        
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side='left',
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                trust_remote_code=True
            ).to(self.device).eval()
            
            # 获取yes和no的token ID
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            
            print("模型加载成功！")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    def _format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """
        构建符合模型要求的输入格式
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n"
    
    def _process_inputs(self, pairs: list[str]) -> dict:
        """
        对格式化的文本对进行分词和填充
        
        Args:
            pairs: 格式化的查询-文档对列表
            
        Returns:
            处理后的模型输入（已移动到指定设备）
        """
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        return inputs.to(self.model.device)
    
    @torch.no_grad()
    def _compute_scores(self, pairs: list[str]) -> list[float]:
        """
        计算所有查询-文档对的相关性分数
        
        Args:
            pairs: 格式化的查询-文档对列表
            
        Returns:
            每个文档的相关性分数（yes的概率）
        """
        inputs = self._process_inputs(pairs)
        
        # 获取最后一个token的logits
        last_token_logits = self.model(** inputs).logits[:, -1, :]
        
        # 提取"yes"和"no"的logit值
        true_logits = last_token_logits[:, self.token_true_id]
        false_logits = last_token_logits[:, self.token_false_id]
        
        # 计算yes的概率作为分数
        scores_tensor = torch.stack([false_logits, true_logits], dim=1)
        probabilities = torch.nn.functional.softmax(scores_tensor, dim=1)
        
        return probabilities[:, 1].cpu().tolist()
    
    def run(self, query: str, docs: list[str], instruction: str = None, with_score: str = False):
        """
        对文档进行重排。返回排序后的文档列表
        """
        # 为每一个doc和query都构建查询-文档对
        pairs = [self._format_instruction(instruction, query, doc) for doc in docs]
        
        # 计算相关性分数
        scores = self._compute_scores(pairs)
        
        # 排序并返回结果
        results = list(zip(docs, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        if with_score:
            # 打印结果
            print(f"查询: {query}")
            print("重排结果:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"{i}. 分数: {score:.4f} | 文档: {doc}")
            return results
        else:
            results_docs = [r[0] for r in results]
            # 打印结果
            print(f"查询: {query}")
            print("重排结果:")
            for doc in results_doc:
                print(f"文档: {doc}")
            return results_docs


# 使用示例
if __name__ == "__main__":
    # 初始化重排器
    reranker = Reranker(model_path='./model')
    
    # 测试示例
    test_query = "日本最高的山是哪座？"
    test_docs = [
        "富士山是日本的最高峰，海拔3776米。",
        "日本是一个多山的国家，拥有许多著名的山脉，如日本阿尔卑斯山脉。",
        "东京是日本的首都和最大的城市。",
    ]
    
    # 运行重排
    ranked_results = reranker.run(test_query, test_docs, with_score=False)
    

    # 其它经典测试
    # 测试 1: 基本相关性
    # query1 = "日本最高的山是哪座？"
    # docs1 = [
    #     "富士山是日本的最高峰，海拔3776米。",
    #     "日本是一个多山的国家，拥有许多著名的山脉，如日本阿尔卑斯山脉。",
    #     "东京是日本的首都和最大的城市。",
    # ]
    
    # 测试 2: 语义理解 vs. 关键词匹配
    # query2 = "我应该如何应对全球变暖？"
    # docs2 = [
    #     "减少碳足迹，例如选择公共交通和使用节能电器，是缓解气候变化的关键。",
    #     "这部关于“全球旅行”的电影非常“温暖”人心。",
    #     "全球变暖导致海平面上升和极端天气事件频发。",
    # ]

    # 测试 3: 对细微差别的识别
    # query3 = "苹果公司出品的最好的笔记本电脑是哪款？"
    # docs3 = [
    #     "MacBook Pro以其强大的性能和出色的显示效果，被广泛认为是专业人士的首选。",
    #     "MacBook Air是苹果最轻薄、性价比最高的笔记本系列。",
    #     "微软的Surface Laptop在Windows阵营中是一款优秀的高端笔记本。",
    # ]

    # 测试 4: 抗干扰能力
    # query4 = "如何学习Python编程？"
    # docs4 = [
    #     "初学者可以从官方教程开始，然后通过编写小程序来实践。",
    #     "学习Python，Python编程，Python教程，免费Python，快速学习Python编程语言。",
    #     "Python是一种广泛应用于数据科学、Web开发和人工智能的流行语言。",
    # ]

    # 测试 5: 跨语言能力
    # query5 = "中国的首都在哪里？"
    # docs5 = [
    #    "Beijing is the capital of the People's Republic of China.",
    #    "Shanghai is one of the largest cities in China.",
    #    "The Great Wall is a famous landmark in China.",
    # ]
