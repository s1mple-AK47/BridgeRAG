from neo4j import Driver
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def upsert_graph_structure(driver: Driver, doc_id: str, doc_summary: str, chunks: List[Dict[str, Any]], entities: List[Dict[str, Any]], batch_size: int = 500):
    """
    以批量方式向 Neo4j 中一次性插入或更新一个文档的完整图结构。

    该函数在一个事务中完成以下操作：
    1. 创建或更新 Document 节点，并设置其 summary。
    2. 为该文档创建所有 Chunk 节点。
    3. 为该文档创建所有 Entity 节点。
    4. 创建从 Entity 到其来源 Chunk 的 :SOURCED_FROM 关系。
    5. 创建从 Chunk 到其所属 Document 的 :PART_OF 关系。

    参数:
        driver (Driver): Neo4j 数据库驱动实例。
        doc_id (str): 文档的唯一标识符。
        doc_summary (str): 文档的摘要。
        chunks (List[Dict]): 该文档的分块列表。
        entities (List[Dict]): 该文档的实体列表。
        batch_size (int): 此参数保留用于未来可能的批处理优化，当前未使用。
    """
    logger.info(f"准备为文档 '{doc_id}' 批量更新/插入图结构...")

    query = """
    // 1. 创建 Document 节点并设置摘要
    MERGE (d:Document {doc_id: $doc_id})
    SET d.summary = $doc_summary

    // 2. 批量创建 Chunk 节点并连接到 Document
    WITH d
    UNWIND $chunks as chunk_props
    MERGE (c:Chunk {chunk_id: chunk_props.chunk_id})
    SET c.content = chunk_props.content
    MERGE (c)-[:PART_OF]->(d)

    // 3. 批量创建 Entity 节点
    WITH d
    UNWIND $entities as entity_props
    MERGE (e:Entity {entity_id: entity_props.entity_id})
    SET e.name = entity_props.name,
        e.summary = entity_props.summary,
        e.doc_id = entity_props.doc_id,
        e.is_named_entity = entity_props.is_named_entity,
        e.type = entity_props.type
    
    // 4. 展开每个实体的来源 chunk_id 列表，并创建关系
    WITH e, entity_props
    UNWIND entity_props.source_chunk_ids as chunk_id
    MATCH (c:Chunk {chunk_id: chunk_id})
    MERGE (e)-[:SOURCED_FROM]->(c)
    MERGE (c)-[:MENTIONS]->(e)
    """
    
    params = {
        "doc_id": doc_id,
        "doc_summary": doc_summary,
        "chunks": chunks,
        "entities": entities
    }
    
    try:
        with driver.session() as session:
            session.run(query, **params)
        logger.info(f"成功为文档 '{doc_id}' 更新了图结构。")
    except Exception as e:
        logger.error(f"为文档 '{doc_id}' 更新图结构时发生错误: {e}")
        raise


def upsert_relations(driver: Driver, relations: List[Dict[str, Any]], batch_size: int = 500):
    """
    以批量方式向 Neo4j 中插入或更新关系。

    该函数首先根据 `entity_id` 匹配源节点和目标节点，
    然后使用 MERGE 创建它们之间的关系。

    参数:
        driver (Driver): Neo4j 数据库驱动实例。
        relations (List[Dict[str, Any]]): 需要被存入的关系列表。
        batch_size (int): 每次向数据库发送的关系数量。
    """
    if not relations:
        logger.info("关系列表为空，无需操作。")
        return

    logger.info(f"准备向 Neo4j 中批量更新/插入 {len(relations)} 个关系...")

    query = """
    UNWIND $batch as relation
    MATCH (source:Entity {entity_id: relation.source_entity_id})
    MATCH (target:Entity {entity_id: relation.target_entity_id})
    MERGE (source)-[r:RELATED_TO]->(target)
    SET r.description = relation.description,
        r.strength = relation.strength,
        r.keywords = relation.keywords
    """

    for i in range(0, len(relations), batch_size):
        batch = relations[i:i + batch_size]
        try:
            with driver.session() as session:
                session.run(query, batch=batch)
            logger.info(f"成功处理了 {len(batch)} 个关系的批次 (批次 {i // batch_size + 1})。")
        except Exception as e:
            logger.error(f"处理关系批次时发生错误: {e}")
            raise

def delete_entities(driver: Driver, doc_id: str, entity_name: Optional[str] = None):
    """
    根据文档 ID 和可选的实体名称删除实体及其关系。

    - 如果只提供了 `doc_id`，则删除该文档下的所有实体及其关系。
    - 如果同时提供了 `doc_id` 和 `entity_name`，则只删除指定的那个实体及其关系。

    该函数使用 `DETACH DELETE` 来保证原子性。

    参数:
        driver (Driver): Neo4j 数据库驱动实例。
        doc_id (str): 需要删除实体的文档 ID。
        entity_name (Optional[str], optional): 需要删除的特定实体的名称。默认为 None。
    """
    params = {"doc_id": doc_id}
    if entity_name:
        logger.info(f"准备删除实体 (doc_id='{doc_id}', entity_name='{entity_name}') 及其关系...")
        query = "MATCH (e:Entity {doc_id: $doc_id, entity_name: $entity_name}) DETACH DELETE e"
        params["entity_name"] = entity_name
    else:
        logger.info(f"准备删除 doc_id='{doc_id}' 的所有实体和关系...")
        query = "MATCH (e:Entity {doc_id: $doc_id}) DETACH DELETE e"

    try:
        with driver.session() as session:
            result = session.run(query, **params)
            summary = result.consume()
            nodes_deleted = summary.counters.nodes_deleted
            relationships_deleted = summary.counters.relationships_deleted
            
            if entity_name:
                log_msg = f"成功删除了实体 (doc_id='{doc_id}', entity_name='{entity_name}')。"
            else:
                log_msg = f"成功删除了 doc_id='{doc_id}' 的所有数据。"
            logger.info(f"{log_msg} 删除了 {nodes_deleted} 个节点和 {relationships_deleted} 个关系。")

    except Exception as e:
        log_context = f"doc_id='{doc_id}'"
        if entity_name:
            log_context += f", entity_name='{entity_name}'"
        logger.error(f"删除实体时发生错误 ({log_context}): {e}")
        raise

def query_graph(driver: Driver, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    执行一个只读的 Cypher 查询并以字典列表的形式返回结果。

    参数:
        driver (Driver): Neo4j 数据库驱动实例。
        query (str): 要执行的 Cypher 查询语句。
        params (Dict[str, Any], optional): 查询语句中的参数。默认为 None。

    返回:
        List[Dict[str, Any]]: 查询结果的列表，每个结果是一个字典。
    """
    if params is None:
        params = {}
    
    logger.info(f"准备执行查询: {query}")
    try:
        with driver.session() as session:
            # 使用 execute_read 来确保事务是只读的
            def _execute_query(tx):
                result = tx.run(query, params)
                return [record.data() for record in result]

            results = session.execute_read(_execute_query)
        logger.info(f"成功执行查询并返回 {len(results)} 条记录。")
        return results
    except Exception as e:
        logger.error(f"执行查询时发生错误: {e}")
        raise 

def create_entity_index(driver: Driver):
    """
    为实体节点的 `doc_id` 和 `entity_name` 属性创建复合索引。

    索引可以显著提高基于这些属性查找节点的速度。
    该操作是幂等的，如果索引已存在，则不会重复创建。

    参数:
        driver (Driver): Neo4j 数据库驱动实例。
    """
    query = "CREATE INDEX entity_doc_id_name_index IF NOT EXISTS FOR (e:Entity) ON (e.doc_id, e.entity_name)"
    logger.info("准备为 (Entity) 的 (doc_id, entity_name) 创建复合索引...")
    try:
        with driver.session() as session:
            session.run(query)
        logger.info("索引创建/验证成功。")
    except Exception as e:
        logger.error(f"创建索引时发生错误: {e}")
        raise 


def create_same_as_links(
    driver: Driver, links: List[Dict[str, Any]], batch_size: int = 500
) -> None:
    """
    以批量方式在实体之间创建 SAME_AS 关系。

    参数:
        driver: Neo4j 数据库驱动实例。
        links: 一个字典列表，其中每个字典代表一个链接，
               并包含 'entity_1_id', 'entity_2_id', 和 'score'。
        batch_size: 每个批次处理的链接数量。
    """
    if not links:
        logger.info("没有需要创建的 SAME_AS 链接。")
        return

    query = """
    UNWIND $links as link
    MATCH (a:Entity {entity_id: link.entity_1_id})
    MATCH (b:Entity {entity_id: link.entity_2_id})
    MERGE (a)-[r:SAME_AS]->(b)
    ON CREATE SET r.score = link.score, r.created_at = timestamp()
    ON MATCH SET r.score = link.score, r.updated_at = timestamp()
    """

    total_links = len(links)
    logger.info(f"准备开始分批创建 {total_links} 个 SAME_AS 链接...")

    for i in range(0, total_links, batch_size):
        batch = links[i : i + batch_size]
        try:
            with driver.session() as session:
                session.run(query, links=batch)
            logger.info(f"成功处理批次 {i//batch_size + 1}，包含 {len(batch)} 个链接。")
        except Exception as e:
            logger.error(
                f"为索引为 {i} 的批次创建 SAME_AS 链接失败: {e}"
            )
            # 根据需求，您可能希望重新抛出异常或进行更优雅的处理
            raise

    logger.info(f"成功创建了总计 {total_links} 个 SAME_AS 链接。") 

def delete_document_graph(tx, doc_id: str):
    """
    删除与特定文档相关的所有图数据：Document 节点、Chunk 节点、Entity 节点以及它们之间的关系。
    """
    query = """
    MATCH (d:Document {doc_id: $doc_id})
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity)
    DETACH DELETE d, c, e
    """
    tx.run(query, doc_id=doc_id)
    logger.info(f"成功删除文档 {doc_id} 的所有相关图数据。")


def get_document_metadata(tx, doc_id: str) -> tuple[str | None, list[str]]:
    """
    根据文档 ID，获取文档的摘要和其包含的所有实体的名称列表。

    参数:
        tx: Neo4j 事务对象。
        doc_id: 文档的唯一标识符。

    返回:
        一个元组，包含 (文档摘要, [实体名称列表])。
        如果未找到文档，则返回 (None, [])。
    """
    query = """
    MATCH (d:Document {doc_id: $doc_id})
    OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity)
    RETURN d.summary AS summary, COLLECT(e.name) AS entities
    """
    result = tx.run(query, doc_id=doc_id).single()
    if result:
        return result["summary"], result["entities"]
    else:
        return None, []


def get_entity_summaries_by_name(tx, doc_id: str, entity_names: list[str]) -> list[dict]:
    """
    在指定文档内，根据实体名称列表获取实体的摘要。

    参数:
        tx: Neo4j 事务对象。
        doc_id: 文档的唯一标识符。
        entity_names: 需要查询其摘要的实体名称列表。

    返回:
        一个字典列表，每个字典包含 'name' 和 'summary'。
        只会返回在文档中且在 entity_names 列表里的实体。
    """
    if not entity_names:
        return []

    query = """
    MATCH (d:Document {doc_id: $doc_id})<-[:PART_OF]-(:Chunk)<-[:SOURCED_FROM]-(e:Entity)
    WHERE e.name IN $entity_names
    RETURN e.name AS name, e.summary AS summary
    """
    results = tx.run(query, doc_id=doc_id, entity_names=entity_names)
    return [record.data() for record in results]


def get_entity_summaries_by_id(tx, entity_ids: list[str]) -> list[dict]:
    """
    根据实体 ID 列表批量获取实体的摘要。

    参数:
        tx: Neo4j 事务对象。
        entity_ids: 需要查询其摘要的实体 ID 列表。

    返回:
        一个字典列表，每个字典包含 'name' 和 'summary'。
    """
    if not entity_ids:
        return []

    query = """
    MATCH (e:Entity)
    WHERE e.entity_id IN $entity_ids
    RETURN e.name AS name, e.summary AS summary
    """
    results = tx.run(query, entity_ids=entity_ids)
    return [record.data() for record in results]


def get_linked_entities(tx, doc_id: str, entity_names: List[str]) -> List[Dict]:
    """
    查找与给定实体（在特定文档内）通过 SAME_AS 关系连接的其他实体。
    这个函数会排除掉那些与源实体属于同一个文档的关联实体。
    """
    if not entity_names:
        return []

    query = """
    MATCH (e1:Entity)
    WHERE e1.doc_id = $doc_id AND e1.name IN $entity_names
    MATCH (e1)-[:SAME_AS]-(e2:Entity)
    WHERE e1.doc_id <> e2.doc_id
    RETURN DISTINCT e2.entity_id AS entity_id, e2.name AS name, e2.summary AS summary
    """
    results = tx.run(query, doc_id=doc_id, entity_names=entity_names)
    return [record.data() for record in results]


def get_documents_metadata_batch(tx, doc_ids: List[str]) -> List[Dict]:
    """
    根据文档ID列表，批量获取每个文档的元数据（摘要和实体列表）。
    """
    query = """
    UNWIND $doc_ids AS docId
    MATCH (d:Document {doc_id: docId})
    // 正确的路径：Document -> Chunk -> Entity
    OPTIONAL MATCH (d)<-[:PART_OF]-(:Chunk)<-[:SOURCED_FROM]-(e:Entity)
    RETURN d.doc_id AS doc_id, d.summary AS summary, COLLECT(DISTINCT e.name) AS entities
    """
    result = tx.run(query, doc_ids=doc_ids)
    records = [record.data() for record in result]
    
    record_map = {record["doc_id"]: record for record in records}
    final_results = []
    for doc_id in doc_ids:
        if doc_id in record_map:
            final_results.append(record_map[doc_id])
        else:
            final_results.append({"doc_id": doc_id, "summary": None, "entities": []})
            logger.warning(f"在批量获取元数据时，未在图中找到文档: {doc_id}")

    return final_results


def get_document_metadata(tx, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    获取单个文档的元数据（摘要和其实体列表）。
    """
    query = """
    MATCH (d:Document {doc_id: $doc_id})
    // 正确的路径：Document -> Chunk -> Entity
    OPTIONAL MATCH (d)<-[:PART_OF]-(:Chunk)<-[:SOURCED_FROM]-(e:Entity)
    RETURN d.doc_id AS doc_id, d.summary AS summary, COLLECT(DISTINCT e.name) AS entities
    """
    result = tx.run(query, doc_id=doc_id).single()
    if result and result["doc_id"] is not None:
        # Neo4j 的 COLLECT 在没有匹配项时会返回一个包含 [null] 的列表，需要清理
        entities = result["entities"]
        cleaned_entities = [e for e in entities if e is not None]
        data = result.data()
        data["entities"] = cleaned_entities
        return data
    else:
        logger.warning(f"在图数据库中未找到文档元数据: {doc_id}")
        return None


def get_documents_by_entities(tx, entity_names: List[str], limit: int) -> List[Dict]:
    """
    根据实体名称列表，查找包含这些实体的文档，并按实体匹配数量排序。
    """
    if not entity_names:
        return []

    query = """
    UNWIND $entity_names AS entityName
    MATCH (e:Entity {name: entityName})
    // 正确的路径：Entity -> Chunk -> Document
    MATCH (e)-[:SOURCED_FROM]->(:Chunk)-[:PART_OF]->(d:Document)
    WITH d, count(DISTINCT e.name) AS overlap_score // 按实体名称去重计数
    ORDER BY overlap_score DESC
    LIMIT $limit
    RETURN d.doc_id AS doc_id, overlap_score AS score
    """
    
    results = tx.run(query, entity_names=entity_names, limit=limit)
    return [record.data() for record in results]

def supplement_knowledge(driver: Driver, doc_id: str, new_entities: List[Dict[str, Any]], new_relations: List[Dict[str, Any]]):
    """
    以增量方式向知识图谱中补充实体和关系。
    - 如果实体已存在，则将新描述追加到现有摘要的末尾。
    - 如果关系已存在，则将新描述追加到现有描述的末尾。
    """
    logger.info(f"准备为文档 '{doc_id}' 增量补充 {len(new_entities)} 个实体和 {len(new_relations)} 个关系...")

    # 补充实体的查询
    entity_query = """
    UNWIND $entities as entity_props
    // 为实体生成全局唯一的 ID
    WITH entity_props, $doc_id + '_' + entity_props.entity_name AS entity_id
    MERGE (e:Entity {entity_id: entity_id})
    ON CREATE SET
        e.name = entity_props.entity_name,
        e.summary = entity_props.description,
        e.doc_id = $doc_id,
        e.is_named_entity = entity_props.is_named_entity,
        e.type = 'named_entity'
    ON MATCH SET
        e.summary = e.summary + '\\n\\n--- (追加描述) ---\\n' + entity_props.description
    // 无论创建还是匹配，都确保实体与来源块连接
    WITH e, entity_props
    MERGE (c:Chunk {chunk_id: entity_props.source_chunk_id})
    MERGE (e)-[:SOURCED_FROM]->(c)
    """

    # 补充关系的查询
    relation_query = """
    UNWIND $relations as rel
    // 匹配源实体和目标实体
    WITH rel, $doc_id + '_' + rel.source_entity AS source_id, $doc_id + '_' + rel.target_entity AS target_id
    MATCH (source:Entity {entity_id: source_id})
    MATCH (target:Entity {entity_id: target_id})
    // 合并关系
    MERGE (source)-[r:RELATED_TO]->(target)
    ON CREATE SET
        r.description = rel.relationship_description
    ON MATCH SET
        r.description = r.description + ' | ' + rel.relationship_description
    """

    try:
        with driver.session() as session:
            if new_entities:
                session.run(entity_query, entities=new_entities, doc_id=doc_id)
                logger.info(f"成功为文档 '{doc_id}' 补充了 {len(new_entities)} 个实体。")
            if new_relations:
                session.run(relation_query, relations=new_relations, doc_id=doc_id)
                logger.info(f"成功为文档 '{doc_id}' 补充了 {len(new_relations)} 个关系。")
    except Exception as e:
        logger.error(f"为文档 '{doc_id}' 增量补充知识时发生错误: {e}")
        raise

def get_knowledge_for_document(driver: Driver, doc_id: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    从 Neo4j 数据库中获取指定文档的所有现有实体和关系。
    这是实现“增量补充”流程的关键函数。

    参数:
        driver (Driver): Neo4j 数据库驱动实例。
        doc_id (str): 需要查询的文档的唯一标识符。

    返回:
        一个字典，包含 'entities' 和 'relations' 两个键，
        其值分别为从数据库中提取的实体和关系列表。
        如果文档不存在或未找到任何相关知识，则返回 None。
    """
    logger.info(f"正在为文档 '{doc_id}' 从 Neo4j 中获取现有知识...")
    
    # 查询1：获取该文档的所有实体
    entity_query = """
    MATCH (e:Entity {doc_id: $doc_id})
    RETURN e as entity
    """
    
    # 查询2：获取该文档内部的所有关系
    relation_query = """
    MATCH (e1:Entity {doc_id: $doc_id})-[r:RELATED_TO]->(e2:Entity {doc_id: $doc_id})
    RETURN 
        e1.entity_id AS source_entity_id,
        e2.entity_id AS target_entity_id,
        e1.name AS source_entity_name,
        e2.name AS target_entity_name,
        r.description AS description,
        r.strength AS strength,
        r.keywords AS keywords,
        e1.doc_id as doc_id
    """

    try:
        with driver.session() as session:
            # 获取实体
            entity_result = session.run(entity_query, doc_id=doc_id)
            # 将 Neo4j 节点对象转换为字典
            entities = [record["entity"]._properties for record in entity_result]

            # 获取关系
            relation_result = session.run(relation_query, doc_id=doc_id)
            relations = [record.data() for record in relation_result]

        if not entities and not relations:
            logger.warning(f"在 Neo4j 中未找到文档 '{doc_id}' 的任何现有实体或关系。")
            return None

        logger.info(f"成功为文档 '{doc_id}' 提取到 {len(entities)} 个实体和 {len(relations)} 个关系。")
        return {"entities": entities, "relations": relations}

    except Exception as e:
        logger.error(f"为文档 '{doc_id}' 获取现有知识时发生错误: {e}")
        raise 