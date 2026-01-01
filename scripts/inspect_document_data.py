import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import Settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.database import vector_ops
from bridgerag.utils.logging_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

def inspect_data(doc_ids: list):
    """
    连接到数据库并检查指定文档ID的数据。
    """
    settings = Settings()
    graph_db = None
    vector_db = None

    try:
        # --- 1. 连接数据库 ---
        logger.info("正在连接到 Neo4j 和 Milvus...")
        graph_db = GraphDBConnection(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        # 修正: 从 milvus_uri 解析主机和端口
        parsed_milvus_uri = urlparse(settings.milvus_uri)
        vector_db = VectorDBConnection(alias="default", host=parsed_milvus_uri.hostname, port=parsed_milvus_uri.port)
        logger.info("数据库连接成功。")

        for doc_id in doc_ids:
            print("\n" + "="*50)
            print(f"Inspecting data for document_id: {doc_id}")
            print("="*50 + "\n")

            # --- 2. 从 Neo4j 查询数据 ---
            with graph_db._driver.session() as session:
                # --- 2a. 首先，做一个最基础的检查：文档节点是否存在 ---
                doc_check_query = """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.summary AS summary
                """
                doc_check_result = session.run(doc_check_query, doc_id=doc_id).single()
                
                print(f"--- Document Node Check in Neo4j ---")
                if doc_check_result:
                    print(f"  SUCCESS: Found Document node for '{doc_id}'.")
                    print(f"  Summary in Neo4j: {doc_check_result['summary']}")
                else:
                    print(f"  FAILURE: Document node for '{doc_id}' NOT found in Neo4j.")
                    # 如果文档节点不存在，后续检查无意义
                    continue
                print("-" * 20 + "\n")

                # --- 2b. 检查结构连接 ---
                chunk_link_query = "MATCH (c:Chunk)-[:PART_OF]->(d:Document {doc_id: $doc_id}) RETURN count(c) AS count"
                chunk_link_count = session.run(chunk_link_query, doc_id=doc_id).single()['count']
                print(f"--- Structural Link Check ---")
                print(f"  Found {chunk_link_count} Chunk node(s) linked to this Document.")
                
                entity_link_query = "MATCH (e:Entity)-[:SOURCED_FROM]->(c:Chunk)-[:PART_OF]->(d:Document {doc_id: $doc_id}) RETURN count(e) AS count"
                entity_link_count = session.run(entity_link_query, doc_id=doc_id).single()['count']
                print(f"  Found {entity_link_count} Entity node(s) linked via Chunks to this Document.")
                print("-" * 20 + "\n")

                # --- 2c. 直接查询与文档相关的实体 (即使是孤儿节点) ---
                # 我们假设 entity_id 的格式是 f"{doc_id}_..."
                entities_query = """
                MATCH (e:Entity) WHERE e.entity_id STARTS WITH $doc_id
                RETURN e.name AS name, e.summary AS summary
                ORDER BY e.name
                """
                entities_result = session.run(entities_query, doc_id=doc_id)
                entities = [record.data() for record in entities_result]
                
                print(f"--- Entities associated with '{doc_id}' (Direct Query) ({len(entities)}) ---")
                if entities:
                    for entity in entities:
                        print(f"  - Name: {entity['name']}\n    Summary: {entity['summary']}")
                else:
                    print("  No associated entities found.")
                print("-" * 20 + "\n")

                # --- 2d. 直接查询实体间的关系 ---
                relations_query = """
                MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
                WHERE e1.entity_id STARTS WITH $doc_id AND e2.entity_id STARTS WITH $doc_id
                RETURN e1.name AS source, e2.name AS target, r.description AS description, r.strength AS strength
                ORDER BY source, target
                """
                relations_result = session.run(relations_query, doc_id=doc_id)
                relations = [record.data() for record in relations_result]

                print(f"--- Relations between associated entities ({len(relations)}) ---")
                if relations:
                    for rel in relations:
                        print(f"  - ({rel['source']}) -> [{rel['description']} (Strength: {rel['strength']})] -> ({rel['target']})")
                else:
                    print("  No relations found.")
                print("-" * 20 + "\n")


            # --- 3. 从 Milvus 查询文档摘要 ---
            summary_collection_name = settings.summary_collection_name
            pk_field = vector_ops.get_pk_field(summary_collection_name, "default")
            
            summary_result = vector_ops.query_by_filter(
                collection_name=summary_collection_name,
                filter_expr=f"{pk_field} in ['{doc_id}']",
                output_fields=["summary"]
            )
            
            print(f"--- Document Summary in Milvus ---")
            if summary_result and 'summary' in summary_result[0]:
                print(summary_result[0]['summary'])
            else:
                print("  No summary found.")
            print("-" * 20 + "\n")


    except Exception as e:
        logger.error(f"检查数据时发生错误: {e}", exc_info=True)
    finally:
        if graph_db:
            graph_db.close()
        if vector_db:
            vector_db.close()
        logger.info("数据库连接已关闭。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect document data in Neo4j and Milvus.")
    parser.add_argument(
        "doc_ids",
        nargs='+',
        help="One or more document IDs to inspect (e.g., george_lucas star_wars)."
    )
    args = parser.parse_args()
    
    inspect_data(args.doc_ids)
