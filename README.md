# BridgeRAG

A knowledge graph-based Retrieval-Augmented Generation system for multi-hop question answering.

## Architecture Overview

```
Offline: Documents → Chunking → Entity/Relation Extraction → Knowledge Fusion → Storage → Entity Linking
Online:  Question → Routing → Reasoning → Retrieval → Synthesis → Answer
```

## Dependencies

- Neo4j: Knowledge graph storage
- Milvus: Vector database
- MinIO: Raw document storage
- vLLM: Local LLM inference service

## Quick Start

### 1. Start Services

```bash
docker-compose up -d
```

### 2. Configure Environment

Copy and edit configuration files:
```bash
cp .env.example .env
# Edit .env to set database connection info

# Edit configs/config.yaml to set model paths, etc.
```

### 3. Initialize Databases

```bash
python scripts/initialize_databases.py
```

### 4. Prepare Data

Data format (`hotpot_docs.jsonl`):
```json
{"id": "document_title", "text": "document_content"}
```

For HotpotQA dataset, convert format first:
```bash
python scripts/convert_hotpot_data.py
```

### 5. Run Offline Pipeline

```bash
python scripts/run_offline_pipeline_multiprocess.py
```

Supports checkpoint resume - rerunning will skip already processed documents.

### 6. Entity Linking (Optional)

After offline processing, run entity linking to establish cross-document entity associations:
```bash
python scripts/run_entity_linking.py
```

### 7. Online Query

Single query test:
```bash
python scripts/run_online_query.py
```

Batch evaluation:
```bash
python scripts/run_online_query_benchmark_hotpot.py
```

## Configuration

### configs/config.yaml

```yaml
llm:
  llm_api_type: "vllm"           # vllm or siliconflow
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

## Scripts Reference

| Script | Description |
|--------|-------------|
| `initialize_databases.py` | Initialize Milvus collections and Neo4j indexes |
| `convert_hotpot_data.py` | Convert HotpotQA data format |
| `run_offline_pipeline_multiprocess.py` | Multi-process offline processing pipeline |
| `run_entity_linking.py` | Cross-document entity linking |
| `run_online_query.py` | Single online query |
| `run_online_query_benchmark_hotpot.py` | Batch query evaluation |
| `sync_log_from_milvus.py` | Sync processed document log from Milvus |

## Performance Tuning

### Offline Processing Concurrency

Edit `scripts/run_offline_pipeline_multiprocess.py`:
```python
NUM_WORKERS = 4  # Adjust based on CPU/GPU resources
```

### Neo4j Write Optimization

Ensure indexes are created (`initialize_databases.py` creates them automatically):
```cypher
CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_id);
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id);
CREATE INDEX doc_id_index IF NOT EXISTS FOR (d:Document) ON (d.doc_id);
```
