import logging
import json
import os
from typing import List, Dict, Any, TYPE_CHECKING
from io import BytesIO

# 导入在运行时实际需要的模块
from bridgerag.database import graph_ops, vector_ops, object_storage_ops

# 使用 TYPE_CHECKING 来避免循环导入问题，同时为类型检查器提供信息
if TYPE_CHECKING:
    from neo4j import Driver
    from minio import Minio
    from bridgerag.core.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


def _delete_partition_data(
    doc_id: str,
    neo4j_driver: "Driver",
    minio_client: "Minio",
    minio_bucket_name: str,
    milvus_collection_names: Dict[str, str],
    milvus_alias: str,
):
    """
    在插入新数据前，删除指定 doc_id 在所有数据库中的旧数据，以确保幂等性。

    参数:
        doc_id (str): 文档的唯一标识符。
        neo4j_driver (Driver): Neo4j 驱动实例。
        minio_client (Minio): Minio 客户端实例。
        minio_bucket_name (str): Minio 存储桶名称。
        milvus_collection_names (Dict[str, str]): 包含所有相关 Milvus 集合名称的字典。
        milvus_alias (str): Milvus 连接别名。
    """
    logger.warning(f"开始为文档 '{doc_id}' 清理旧数据...")

    # 1. 从 Neo4j 删除
    try:
        # 一个更健壮的方法是删除与 doc_id 相关的所有节点和关系
        # 注意：这会删除 Document, Chunks, Entities
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        OPTIONAL MATCH (c)<-[:SOURCED_FROM]-(e:Entity)
        DETACH DELETE d, c, e
        """
        with neo4j_driver.session() as session:
            session.run(query, doc_id=doc_id)
        logger.info(f"成功从 Neo4j 中清除了文档 '{doc_id}' 的图数据。")
    except Exception as e:
        logger.error(f"从 Neo4j 清理文档 '{doc_id}' 的数据时出错: {e}")
        raise

    # 2. 从 Milvus 删除
    try:
        chunk_collection_name = milvus_collection_names.get('chunk_collection_name')
        entity_collection_name = milvus_collection_names.get('entity_collection_name')
        summary_collection_name = milvus_collection_names.get('summary_collection_name')

        # 在执行删除前，先检查是否有数据存在，避免对空集合或未就绪的索引执行 load 操作
        def check_and_delete(collection_name: str, filter_expr: str):
            try:
                # 检查是否有数据，避免在空集合上加载索引的开销
                count_res = vector_ops.query_by_filter(collection_name, filter_expr, output_fields=["count(*)"])
                if count_res and count_res[0]["count(*)"] > 0:
                    vector_ops.delete_by_filter(collection_name, filter_expr)
                    logger.info(f"从集合 '{collection_name}' 中删除了与 '{doc_id}' 相关的旧条目。")
            except Exception as e:
                logger.error(f"在集合 '{collection_name}' 中为 doc_id '{doc_id}' 执行检查和删除操作时出错: {e}")
                raise

        # 使用双引号来包裹 doc_id，以处理 doc_id 本身包含单引号的情况
        check_and_delete(chunk_collection_name, f'doc_id == "{doc_id}"')
        check_and_delete(entity_collection_name, f'doc_id == "{doc_id}"')
        check_and_delete(summary_collection_name, f'doc_id == "{doc_id}"')

        logger.info(f"成功从 Milvus 中清除了文档 '{doc_id}' 的向量数据。")
    except Exception as e:
        logger.error(f"从 Milvus 清理文档 '{doc_id}' 的数据时出错: {e}")
        raise

    # 3. 从 MinIO 删除
    try:
        object_name = doc_id
        object_storage_ops.delete_object(minio_client, minio_bucket_name, object_name)
        logger.info(f"成功从 Minio 中清除了文档 '{doc_id}' 的处理结果备份。")
    except Exception as e:
        logger.error(f"从 Minio 清理文档 '{doc_id}' 的数据时出错: {e}")
        raise
    
    logger.info(f"文档 '{doc_id}' 的旧数据清理完毕。")


def save_partition(
    doc_id: str,
    chunks: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    document_summary: str,
    # 数据库和服务的客户端
    neo4j_driver: "Driver",
    minio_client: "Minio",
    embedding_client: "EmbeddingClient",
    milvus_alias: str,
    # 配置信息
    minio_configs: Dict[str, str],
    embedding_configs: Dict[str, str],
    milvus_collection_names: Dict[str, str],
    force_rewrite: bool = False
):
    """
    将单个文档的处理结果（知识分区）保存到所有持久化存储中。

    这个函数是数据持久化的核心调度器，它执行以下操作：
    1. (可选) 调用 _delete_partition_data 清理旧数据，确保操作的幂等性。
    2. 调用 EmbeddingClient 为文本数据（块、实体描述、文档摘要）生成向量。
    3. 调用 graph_ops 将实体和关系存入 Neo4j。
    4. 调用 vector_ops 将块、实体和摘要的向量存入 Milvus。
    5. 调用 object_storage_ops 将完整的处理结果（JSON格式）备份到 MinIO。

    参数:
        doc_id (str): 文档的唯一标识符。
        chunks (List[Dict]): 文档分块列表。
        entities (List[Dict]): 提取出的实体列表。
        relations (List[Dict]): 提取出的关系列表。
        document_summary (str): 生成的文档摘要。
        neo4j_driver (Driver): Neo4j 驱动实例。
        minio_client (Minio): Minio 客户端实例。
        embedding_client (EmbeddingClient): 嵌入模型客户端。
        milvus_alias (str): Milvus 连接别名。
        minio_configs (Dict): Minio 相关配置, e.g., {"bucket_name": "bridgerag"}.
        embedding_configs (Dict): 嵌入模型相关配置, e.g., {"embedding_model": "nomic-embed-text-v1"}.
        force_rewrite (bool): 是否强制重写。如果为 True，将先删除旧数据。
    """
    logger.info(f"开始为文档 '{doc_id}' 保存知识分区...")

    if force_rewrite:
        _delete_partition_data(
            doc_id, neo4j_driver, minio_client, minio_configs["bucket_name"], milvus_collection_names, milvus_alias
        )

    # --- 1. 向量化数据 ---
    logger.info("开始为所有文本数据生成向量...")
    # 准备需要向量化的文本
    chunk_texts = [chunk['content'] for chunk in chunks]
    # 为实体描述和文档摘要生成向量
    entity_desc_texts = [entity['summary'] for entity in entities]
    summary_texts = [document_summary]
    
    # 获取向量
    chunk_embeddings = embedding_client.get_embeddings(chunk_texts)
    entity_embeddings = embedding_client.get_embeddings(entity_desc_texts)
    summary_embedding = embedding_client.get_embeddings(summary_texts)[0]

    # 将向量填充回原数据结构
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = chunk_embeddings[i]

    for i, entity in enumerate(entities):
        entity["embedding"] = entity_embeddings[i]
    
    logger.info("向量生成完毕。")
    
    # --- 2. 准备并保存到 Milvus ---
    logger.info("准备向 Milvus 中保存向量数据...")
    try:
        # 准备块数据
        chunk_data_for_milvus = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": doc_id,
                "content": chunk["content"],
                "dense_vector": chunk["embedding"],
            }
            for chunk in chunks
        ]
        vector_ops.upsert_vectors(milvus_collection_names['chunk_collection_name'], chunk_data_for_milvus, milvus_alias)

        # 准备实体数据
        entity_data_for_milvus = [
            {
                "entity_id": entity['entity_id'],
                "doc_id": doc_id,
                "name": entity["name"],
                "is_named_entity": entity["is_named_entity"],
                "description": entity["summary"],
                "dense_vector": entity["embedding"],
            }
            for entity in entities
        ]
        if entity_data_for_milvus:
            vector_ops.upsert_vectors(milvus_collection_names['entity_collection_name'], entity_data_for_milvus, milvus_alias)
        else:
            logger.warning(f"文档 '{doc_id}' 没有提取到任何实体，跳过向 Milvus 实体集合的插入操作。")

        # 准备摘要数据
        summary_data_for_milvus = [
            {
                "doc_id": doc_id,
                "summary": document_summary,
                "dense_vector": summary_embedding,
            }
        ]
        vector_ops.upsert_vectors(milvus_collection_names['summary_collection_name'], summary_data_for_milvus, milvus_alias)
        logger.info("成功将所有向量数据保存到 Milvus。")
    except Exception as e:
        logger.error(f"保存数据到 Milvus 时出错: {e}")
        raise

    # --- 3. 保存到 Neo4j ---
    logger.info("准备向 Neo4j 中保存图数据...")
    try:
        # 为了遵循最佳实践，不将向量存入 Neo4j，创建一个不含 embedding 的实体副本列表
        entities_for_neo4j = [
            {k: v for k, v in entity.items() if k != 'embedding'}
            for entity in entities
        ]

        # 使用新的统一函数来创建文档、块、实体和它们之间的关系
        graph_ops.upsert_graph_structure(
            driver=neo4j_driver,
            doc_id=doc_id,
            doc_summary=document_summary,
            chunks=[{"chunk_id": c["chunk_id"], "content": c["content"]} for c in chunks],
            entities=entities_for_neo4j  # 传递不含 embedding 的实体列表
        )
        
        # 准备关系数据
        relations_for_neo4j = [
            {
                "source_entity_id": rel["source_entity_id"],
                "target_entity_id": rel["target_entity_id"],
                "description": rel.get("description"),
                "strength": rel.get("strength", 5.0),
                "keywords": rel.get("keywords", [])
            } for rel in relations
        ]
        # 修正: 移除 driver 参数
        graph_ops.upsert_relations(neo4j_driver, relations_for_neo4j)
        logger.info("成功将图数据保存到 Neo4j。")
    except Exception as e:
        logger.error(f"保存数据到 Neo4j 时出错: {e}")
        raise
        
    # --- 4. 备份到 MinIO ---
    logger.info("准备将处理结果备份到 MinIO...")
    try:
        # 创建一个包含所有信息的字典用于备份
        backup_data = {
            "doc_id": doc_id,
            "document_summary": document_summary,
            "entities": entities,
            "relations": relations,
            "chunks": chunks
        }
        # 移除备份数据中的向量，减小文件大小
        for item_list in [backup_data["entities"], backup_data["chunks"]]:
            for item in item_list:
                item.pop("embedding", None)

        json_content = json.dumps(backup_data, indent=4, ensure_ascii=False)
        object_name = doc_id
        
        object_storage_ops.upload_text_as_object(
            client=minio_client,
            bucket_name=minio_configs["bucket_name"],
            object_name=object_name,
            content=json_content,
            content_type="application/json; charset=utf-8"
        )
        logger.info("成功将处理结果备份到 MinIO。")
    except Exception as e:
        logger.error(f"备份处理结果到 MinIO 时出错: {e}")
        raise

    # --- 5. 记录成功处理的文档 ---
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        success_log_path = os.path.join(log_dir, "successful_documents.log")
        with open(success_log_path, "a", encoding="utf-8") as f:
            f.write(f"{doc_id}\n")
    except Exception as e:
        logger.error(f"记录成功文档 '{doc_id}' 到日志文件时出错: {e}")

    logger.info(f"文档 '{doc_id}' 的知识分区已成功保存到所有数据库。") 