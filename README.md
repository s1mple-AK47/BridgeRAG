# BridgeRAG

基于知识图谱的检索增强生成系统，支持多跳问答。

## 架构概览

```
离线阶段: 文档 → 分块 → 实体/关系提取 → 知识融合 → 存储 → 实体链接
在线阶段: 问题 → 路由 → 推理 → 检索 → 综合 → 答案
```

## 依赖服务

- Neo4j: 知识图谱存储
- Milvus: 向量数据库
- MinIO: 原始文档存储
- vLLM: 本地大模型推理服务

## 快速开始

### 1. 启动依赖服务

```bash
docker-compose up -d
```

### 2. 配置环境

复制并编辑配置文件：
```bash
cp .env.example .env
# 编辑 .env 设置数据库连接信息

# 编辑 configs/config.yaml 设置模型路径等
```

### 3. 初始化数据库

```bash
python scripts/initialize_databases.py
```

### 4. 准备数据

数据格式 (`hotpot_docs.jsonl`):
```json
{"id": "文档标题", "text": "文档内容"}
```

如果使用 HotpotQA 数据集，先转换格式：
```bash
python scripts/convert_hotpot_data.py
```

### 5. 运行离线流水线

```bash
python scripts/run_offline_pipeline_multiprocess.py
```

支持断点续传，中断后重新运行会跳过已处理的文档。

### 6. 实体链接（可选）

在离线处理完成后，运行实体链接以建立跨文档的实体关联：
```bash
python scripts/run_entity_linking.py
```

### 7. 在线查询

单条查询测试：
```bash
python scripts/run_online_query.py
```

批量评测：
```bash
python scripts/run_online_query_benchmark_hotpot.py
```

## 配置说明

### configs/config.yaml

```yaml
llm:
  llm_api_type: "vllm"           # vllm 或 siliconflow
  vllm_host: "localhost"
  vllm_port: 8989
  vllm_generation_model_name: "Qwen2.5-14B"
  vllm_tokenizer_path: "/path/to/tokenizer"
  
database:
  chunk_collection_name: "bridgerag_chunks"
  entity_collection_name: "bridgerag_entities"
  summary_collection_name: "bridgerag_summaries"
```

### .env

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

MILVUS_URI=http://localhost:19530
```

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `initialize_databases.py` | 初始化 Milvus 集合和 Neo4j 索引 |
| `convert_hotpot_data.py` | 转换 HotpotQA 数据格式 |
| `run_offline_pipeline_multiprocess.py` | 多进程离线处理流水线 |
| `run_entity_linking.py` | 跨文档实体链接 |
| `run_online_query.py` | 单条在线查询 |
| `run_online_query_benchmark_hotpot.py` | 批量查询评测 |
| `sync_log_from_milvus.py` | 从 Milvus 同步已处理文档日志 |

## 性能调优

### 离线处理并发数

编辑 `scripts/run_offline_pipeline_multiprocess.py`:
```python
NUM_WORKERS = 4  # 根据 CPU/GPU 资源调整
```

### Neo4j 写入优化

确保已创建索引（`initialize_databases.py` 会自动创建）：
```cypher
CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_id);
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id);
CREATE INDEX doc_id_index IF NOT EXISTS FOR (d:Document) ON (d.doc_id);
```
