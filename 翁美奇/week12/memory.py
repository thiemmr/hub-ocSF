"""
    用于存储和检索 Agent 的记忆信息，支持两种模式：
    1. faiss 模式：使用 FAISS 向量索引进行语义检索，适合长期记忆和大规模数据
    记忆信息以向量形式存储，便于与用户输入进行相似度计算和检索。每条记忆信息包含文本内容和对应的向量
    记忆信息的写入和读取操作需要根据模式进行不同的处理：
    - 写入操作：
        - faiss 模式：将文本内容转换为向量，并添加到 FAISS 索引中，同时保存文本内容到元数据列表
    - 读取操作：
        - faiss 模式：将查询文本转换为向量，并在 FAISS 索引中进行相似度检索，返回最相似的记忆信息
    通过这种方式，Agent 可以根据用户的输入和历史记忆信息进行更智能的响应和决策，提升交互体验和准确性。
"""
import faiss
import numpy as np
import time
import os

from pathlib import Path
import json
import logging
import re

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
MEMORY_DIR      = BASE_DIR / 'memory'
DIALOGUE_INDEX_PATH = MEMORY_DIR / "dialogue_index.bin"
DIALOGUE_META_PATH  = MEMORY_DIR / "dialogue_meta.json"
PREFERENCE_META_PATH = MEMORY_DIR / "preference_meta.json"

class AgentMemory:
    def __init__(self, mode='faiss', dim=1536, max_context_len=10):
        self.mode = mode
        self.max_context_len = max_context_len
        self.dim = dim
        self.history_message = []
        self.context_buffer = []
        self.preferences = []
        
        self.embedding_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[Memory] 初始化完成，模式: %s, 维度: %d", mode, dim)

        self.load_all()

    def load_all(self):
        if DIALOGUE_INDEX_PATH.exists() and DIALOGUE_META_PATH.exists():
            self.dialogue_index = faiss.read_index(str(DIALOGUE_INDEX_PATH))
            with open(DIALOGUE_META_PATH, 'r', encoding='utf-8') as f:
                self.history_message = json.load(f)
            logger.info("[Memory] 加载对话索引: %d 条记录", len(self.history_message))
        else:
            self.dialogue_index = faiss.IndexFlatL2(self.dim)
            self.history_message = []
            logger.info("[Memory] 初始化新对话索引")

        if PREFERENCE_META_PATH.exists():
            with open(PREFERENCE_META_PATH, 'r', encoding='utf-8') as f:
                self.preferences = json.load(f)
            logger.info("[Memory] 加载用户偏好: %d 条", len(self.preferences))
        else:
            self.preferences = []
            logger.info("[Memory] 初始化新偏好列表")

    def save_all(self):
        faiss.write_index(self.dialogue_index, str(DIALOGUE_INDEX_PATH))
        with open(DIALOGUE_META_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.history_message, f, ensure_ascii=False, indent=4)
        with open(PREFERENCE_META_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=4)
        logger.info("[Memory] 保存完成，对话记录: %d, 偏好: %d", len(self.history_message), len(self.preferences))

    def retrieve_context(self, query, top_k=3):
        """
        检索与查询相关的上下文。

        返回结构化数据：
            {
                "preferences": ["偏好1", "偏好2"],
                "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            }
        """
        logger.info("[Memory] 检索上下文，查询长度: %d, top_k: %d, 偏好数: %d", len(query), top_k, len(self.preferences))

        history_messages = []
        seen = set()

        # 1. 检索长期记忆（FAISS）
        if self.dialogue_index.ntotal > 0:
            query_vec = self._text_to_vector(query)
            distances, ids = self.dialogue_index.search(query_vec.reshape(1, -1), top_k)

            for doc_id in ids[0]:
                if 0 <= doc_id < len(self.history_message):
                    item = self.history_message[doc_id]
                    if item['role'] != 'system':
                        key = (item['role'], item['content'])
                        if key not in seen:
                            seen.add(key)
                            history_messages.append({"role": item['role'], "content": item['content']})

        # 2. 追加当前会话缓冲区（短期上下文）
        for item in self.context_buffer:
            if item['role'] != 'system':
                key = (item['role'], item['content'])
                if key not in seen:
                    seen.add(key)
                    history_messages.append({"role": item['role'], "content": item['content']})
        
        return {
            "preferences": self.preferences,
            "history": history_messages,
        }

    def write_memory(self, message_content, role):
        doc_id = len(self.history_message)

        record = {
            "id": doc_id,
            "role": role,
            "content": message_content,
            "timestamp": time.time(),
        }
        self.history_message.append(record)

        vector = self._text_to_vector(message_content)
        self.dialogue_index.add(vector.reshape(1, -1))

        self.context_buffer.append({"role": role, "content": message_content})
        if len(self.context_buffer) > self.max_context_len:
            self.context_buffer.pop(0)

        self.save_all()

    def save_preference(self, preference_text):
        logger.info(f"[Memory] save_preference called with: '{preference_text}'")
        if preference_text and preference_text != "NONE":
            if preference_text not in self.preferences:
                self.preferences.append(preference_text)
                try:
                    self.save_all()
                    logger.info("[Memory] 保存新偏好成功: %s", preference_text)
                except Exception as e:
                    logger.error(f"[Memory] 保存偏好失败: {e}")
            else:
                logger.info("[Memory] 偏好已存在，跳过: %s", preference_text)
        else:
            logger.info("[Memory] 无效偏好，跳过")

    def _text_to_vector(self, text):
        try:
            response = self.embedding_client.embeddings.create(
                input=text,
                model="text-embedding-v1"
            )
            vector = response.data[0].embedding
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise