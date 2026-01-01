import logging
from typing import List, Dict, Any, Optional
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    connections,
)
import json # Import the json library

logger = logging.getLogger(__name__)

# --- 1. 集合管理 (Collection Management) ---

def _create_collection_if_not_exists(
    collection_name: str,
    schema: CollectionSchema,
    index_params: List[Dict],
    alias: str = "default",
) -> Collection:
    """
    通用内部函数：如果集合不存在，则创建它，并为其向量字段创建索引。

    参数:
        collection_name (str): 集合的名称。
        schema (CollectionSchema): 集合的 schema 定义。
        index_params (List[Dict]): 向量字段的索引参数列表。
        alias (str): Milvus 连接的别名。

    返回:
        Collection: 创建或加载的 Milvus 集合对象。
    """
    if not utility.has_collection(collection_name, using=alias):
        logger.info(f"集合 '{collection_name}' 不存在，现在开始创建...")
        collection = Collection(
            name=collection_name, schema=schema, using=alias, consistency_level="Strong"
        )
        logger.info(f"集合 '{collection_name}' 创建成功。")
        
        for params in index_params:
            logger.info(f"为字段 '{params['field_name']}' 创建索引...")
            collection.create_index(
                field_name=params['field_name'],
                index_params=params['index_params'],
            )
            logger.info(f"索引创建成功: {params}")
        logger.info(f"集合 '{collection_name}' 的所有索引都已成功创建。")
        return collection
    else:
        logger.info(f"集合 '{collection_name}' 已存在。")
        collection = Collection(collection_name, using=alias)
        # 检查现有集合是否已有索引
        if not collection.has_index():
            logger.warning(f"集合 '{collection_name}' 存在但缺少索引。现在开始创建索引...")
            for params in index_params:
                logger.info(f"为字段 '{params['field_name']}' 创建索引...")
                collection.create_index(
                    field_name=params['field_name'],
                    index_params=params['index_params'],
                )
                logger.info(f"索引创建成功: {params}")
            logger.info(f"集合 '{collection_name}' 的所有索引都已成功创建。")
        else:
            logger.info(f"集合 '{collection_name}' 的索引已存在。")
        return collection

def create_chunk_collection(
    collection_name: str, 
    dense_dim: int, 
    text_max_length: int,
    alias: str = "default"
) -> Collection:
    """为文档块(chunks)创建一个专用的 Milvus 集合。"""
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=text_max_length),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields=fields, description="Collection for document chunks")
    index_params = [
        {
            "field_name": "dense_vector",
            "index_params": {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}},
        }
    ]
    return _create_collection_if_not_exists(collection_name, schema, index_params, alias)

def create_entity_collection(
    collection_name: str, 
    dense_dim: int, 
    text_max_length: int,
    alias: str = "default"
) -> Collection:
    """为文档实体(entities)创建一个专用的 Milvus 集合。"""
    fields = [
        FieldSchema(name="entity_id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="is_named_entity", dtype=DataType.BOOL),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=text_max_length),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields=fields, description="Collection for document entities")
    index_params = [
        {
            "field_name": "dense_vector",
            "index_params": {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}},
        }
    ]
    return _create_collection_if_not_exists(collection_name, schema, index_params, alias)

def create_summary_collection(
    collection_name: str, 
    dense_dim: int, 
    text_max_length: int,
    alias: str = "default"
) -> Collection:
    """为文档摘要(summaries)创建一个专用的 Milvus 集合。"""
    fields = [
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=text_max_length),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields=fields, description="Collection for document summaries")
    index_params = [
        {
            "field_name": "dense_vector",
            "index_params": {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}},
        }
    ]
    return _create_collection_if_not_exists(collection_name, schema, index_params, alias)

def get_pk_field(collection_name: str, alias: str = "default") -> str:
    """获取指定集合的主键字段名称。"""
    collection = Collection(collection_name, using=alias)
    return collection.primary_field.name

def drop_collection(collection_name: str, alias: str = "default"):
    """删除一个集合。"""
    logger.info(f"准备删除集合 '{collection_name}'...")
    utility.drop_collection(collection_name, using=alias)
    logger.info(f"集合 '{collection_name}' 已成功删除。")

# --- 2. 数据操作 (Data Manipulation) ---

def upsert_vectors(collection_name: str, data: List[Dict[str, Any]], alias: str = "default"):
    """批量插入或更新数据到指定的集合。"""
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    logger.info(f"准备向集合 '{collection_name}' 中批量插入/更新 {len(data)} 条数据...")
    mr = collection.upsert(data)
    logger.info(f"数据批量插入/更新成功，主键: {mr.primary_keys}")
    collection.flush()
    logger.info(f"集合 '{collection_name}' flush 完成。")
    return mr

def delete_vectors_by_pks(collection_name: str, pks: List[Any], alias: str = "default"):
    """根据主键列表删除向量。"""
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    expr = f'{collection.primary_field.name} in {pks}'
    logger.info(f"准备从集合 '{collection_name}' 中删除主键为 {pks} 的数据...")
    result = collection.delete(expr)
    logger.info(f"数据删除成功: {result}")
    collection.flush()
    return result

def delete_vectors_by_filter(collection_name: str, filter_expr: str, alias: str = "default"):
    """根据过滤表达式删除向量。"""
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    logger.info(f"准备从集合 '{collection_name}' 中根据过滤条件 '{filter_expr}' 删除数据...")
    # Milvus 的 delete 接口直接接受过滤表达式
    result = collection.delete(filter_expr)
    logger.info(f"数据删除成功: {result}")
    collection.flush()
    return result

def query_by_filter(collection_name: str, filter_expr: str, output_fields: List[str] = None, alias: str = "default") -> List[Dict]:
    """根据过滤表达式查询数据。"""
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    logger.info(f"在集合 '{collection_name}' 中执行查询，过滤条件: '{filter_expr}'...")
    results = collection.query(expr=filter_expr, output_fields=output_fields)
    logger.info(f"查询成功，返回 {len(results)} 条记录。")
    return results

def get_vectors_by_ids(
    collection_name: str,
    ids: List[str],
    id_field: str,
    vector_field: str = "dense_vector",
    alias: str = "default",
    batch_size: int = 10000,
) -> Dict[str, List[float]]:
    """
    根据主键 ID 列表批量检索向量。

    参数:
        collection_name (str): 集合名称。
        ids (List[str]): 要检索的主键 ID 列表。
        id_field (str): 用于查询的 ID 字段名称 (例如 "entity_id")。
        vector_field (str): 向量字段的名称。
        alias (str): Milvus 连接别名。
        batch_size (int): 每批查询的 ID 数量，避免超过 Milvus 限制。

    返回:
        Dict[str, List[float]]: 一个字典，键是主键 ID，值是对应的向量。
    """
    if not ids:
        return {}
    
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    
    logger.info(f"在集合 '{collection_name}' 中根据 {len(ids)} 个 ID 检索向量（批大小: {batch_size}）...")
    
    id_to_vector = {}
    
    # 分批查询
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        
        # 修正: 使用 json.dumps 来安全地格式化ID，以处理特殊字符
        formatted_ids = ", ".join([json.dumps(_id) for _id in batch_ids])
        expr = f'{id_field} in [{formatted_ids}]'
        
        try:
            results = collection.query(
                expr=expr, 
                output_fields=[id_field, vector_field]
            )
            
            for res in results:
                id_to_vector[res[id_field]] = res[vector_field]
            
            logger.debug(f"批次 {i // batch_size + 1}: 检索到 {len(results)} 个向量")
            
        except Exception as e:
            logger.error(f"批次 {i // batch_size + 1} 检索向量时出错: {e}")
            # 继续处理下一批，而不是完全失败
            continue
    
    logger.info(f"成功检索到 {len(id_to_vector)} 个向量。")
    return id_to_vector

# --- 3. 核心搜索 (Core Search) ---

def search(
    collection_name: str,
    dense_vector: List[float],
    sparse_vector, # 通常是字典或 Milvus 的 SparseVector 类型
    top_k: int,
    output_fields: List[str],
    filter_expr: Optional[str] = None,
    alias: str = "default",
) -> List[Dict]:
    """
    执行混合搜索（密集 + 稀疏），并使用 RRF 对结果进行融合重排。
    可以传入一个可选的过滤表达式来预过滤数据。

    参数:
        collection_name (str): 集合名称。
        dense_vector (List[float]): 查询用的密集向量。
        sparse_vector: 查询用的稀疏向量。
        top_k (int): 需要返回的结果数量。
        output_fields (List[str]): 需要返回的字段列表。
        filter_expr (Optional[str], optional): 用于预过滤的表达式，例如 "doc_id == 'some_id'"。默认为 None。
        alias (str, optional): Milvus 连接别名。

    返回:
        List[Dict]: 融合并排序后的结果列表，每个字典包含 'id' 和 'score'。
    """
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    dense_req = {
        "data": [dense_vector], "anns_field": "dense_vector", "param": {"metric_type": "L2", "params": {"ef": 10}}, "limit": top_k
    }
    sparse_req = {
        "data": [sparse_vector], "anns_field": "sparse_vector", "param": {"metric_type": "IP", "params": {}}, "limit": top_k
    }
    
    log_msg = f"在集合 '{collection_name}' 中执行混合搜索..."
    if filter_expr:
        dense_req["expr"] = filter_expr
        sparse_req["expr"] = filter_expr
        log_msg += f" 使用过滤条件: '{filter_expr}'"

    logger.info(log_msg)
    dense_results = collection.search(**dense_req)
    sparse_results = collection.search(**sparse_req)
    
    # RRF 结果融合 (Reciprocal Rank Fusion)
    fused_scores = {}
    k = 60  # RRF 的一个超参数

    # 处理密集搜索结果
    for rank, hit in enumerate(dense_results[0]):
        pk = hit.id
        if pk not in fused_scores:
            fused_scores[pk] = 0
        fused_scores[pk] += 1 / (k + rank + 1)

    # 处理稀疏搜索结果
    for rank, hit in enumerate(sparse_results[0]):
        pk = hit.id
        if pk not in fused_scores:
            fused_scores[pk] = 0
        fused_scores[pk] += 1 / (k + rank + 1)
    
    # 排序并获取 Top-K
    sorted_pks = sorted(fused_scores.keys(), key=lambda pk: fused_scores[pk], reverse=True)[:top_k]

    if not sorted_pks:
        return []

    # 根据排序后的主键查询详细信息
    pk_field = collection.primary_field.name
    res = collection.query(
        expr=f"{pk_field} in {sorted_pks}",
        output_fields=output_fields
    )
    
    # 附加融合分数并按分数重新排序
    final_results = []
    for r in res:
        r['rerank_score'] = fused_scores[r[pk_field]]
        final_results.append(r)
        
    final_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    logger.info(f"混合搜索完成，返回 {len(final_results)} 条融合结果。")
    return final_results


def search_chunks_by_vector(
    collection_name: str,
    query_vector: List[float],
    doc_id: str,
    top_k: int = 2,
    output_fields: List[str] = ["chunk_id", "content"],
    alias: str = "default",
) -> List[Dict]:
    """
    在指定文档内，根据密集向量搜索最相关的文本块。

    参数:
        collection_name (str): 集合名称 (应为块集合)。
        query_vector (List[float]): 查询用的密集向量。
        doc_id (str): 要将搜索范围限定于此文档 ID。
        top_k (int): 需要返回的结果数量。
        output_fields (List[str]): 需要返回的字段列表。
        alias (str): Milvus 连接别名。

    返回:
        List[Dict]: 搜索结果列表，每个字典包含 output_fields 中指定的字段以及 'distance'。
    """
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    search_params = {"metric_type": "L2", "params": {"ef": 10}}
    # 修正：使用双引号来包裹字符串值，以安全地处理 doc_id 中可能包含的单引号
    filter_expr = f'doc_id == "{doc_id}"'

    logger.info(
        f"在集合 '{collection_name}' 中搜索与文档 '{doc_id}' 相关的 top {top_k} 个文本块..."
    )

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )

        # 格式化输出
        formatted_results = []
        for hit in results[0]:
            entity_data = {"distance": hit.distance}
            for field in output_fields:
                entity_data[field] = hit.entity.get(field)
            formatted_results.append(entity_data)

        logger.info(f"搜索成功，找到 {len(formatted_results)} 个文本块。")
        return formatted_results
    except Exception as e:
        logger.error(f"在文档 '{doc_id}' 内搜索文本块时出错: {e}")
        return []


def search_entities_by_vector(
    collection_name: str,
    query_vector: List[float],
    top_k: int,
    output_fields: List[str] = ["entity_id"],
    alias: str = "default",
) -> List[Dict]:
    """
    根据密集向量搜索最相关的实体（无过滤）。
    主要用于实体链接中寻找最近邻。

    参数:
        collection_name (str): 集合名称 (应为实体集合)。
        query_vector (List[float]): 查询用的密集向量。
        top_k (int): 需要返回的结果数量。
        output_fields (List[str]): 需要返回的字段列表。
        alias (str): Milvus 连接别名。

    返回:
        List[Dict]: 搜索结果列表，包含 output_fields 和 'distance'。
    """
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    search_params = {"metric_type": "L2", "params": {"ef": 10}}

    logger.info(
        f"在实体集合 '{collection_name}' 中搜索 top {top_k} 个相关实体..."
    )

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )
        
        # 格式化输出
        formatted_results = []
        for hit in results[0]:
            entity_data = {"distance": hit.distance}
            for field in output_fields:
                entity_data[field] = hit.entity.get(field)
            formatted_results.append(entity_data)

        logger.info(f"实体搜索成功，找到 {len(formatted_results)} 个实体。")
        return formatted_results
    except Exception as e:
        logger.error(f"搜索实体时出错: {e}")
        return []


def search_summaries_by_vector(
    collection_name: str,
    query_vector: List[float],
    top_k: int,
    output_fields: List[str] = ["doc_id"],
    alias: str = "default",
) -> List[Dict]:
    """
    根据密集向量搜索最相关的文档摘要。

    参数:
        collection_name (str): 集合名称 (应为摘要集合)。
        query_vector (List[float]): 查询用的密集向量。
        top_k (int): 需要返回的结果数量。
        output_fields (List[str]): 需要返回的字段列表。
        alias (str): Milvus 连接别名。

    返回:
        List[Dict]: 搜索结果列表，包含 'doc_id' 和 'score'。
    """
    collection = Collection(collection_name, using=alias)
    collection.load() # 确保集合已加载
    search_params = {"metric_type": "L2", "params": {"ef": 10}}

    logger.info(
        f"在摘要集合 '{collection_name}' 中搜索 top {top_k} 个相关文档..."
    )

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )

        formatted_results = [
            {"doc_id": hit.entity.get("doc_id"), "score": hit.distance}
            for hit in results[0]
        ]

        logger.info(f"摘要搜索成功，找到 {len(formatted_results)} 个文档。")
        return formatted_results
    except Exception as e:
        logger.error(f"搜索文档摘要时出错: {e}")
        return []


# --- 4. 内存管理 (Memory Management) ---

def load_collection(collection_name: str, alias: str = "default"):
    """将集合加载到内存中以进行搜索。"""
    collection = Collection(collection_name, using=alias)
    logger.info(f"准备加载集合 '{collection_name}' 到内存...")
    collection.load()
    logger.info("加载完成。")

def release_collection(collection_name: str, alias: str = "default"):
    """从内存中释放集合以节省资源。"""
    collection = Collection(collection_name, using=alias)
    logger.info(f"准备从内存中释放集合 '{collection_name}'...")
    collection.release()
    logger.info("释放完成。") 