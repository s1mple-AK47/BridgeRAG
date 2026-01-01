"""
估算实体链接候选对数量的脚本。

根据 MIN_DOCS 和 MAX_DOCS 参数，统计会生成多少候选对。
"""

import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import Settings
from bridgerag.database.graph_db import GraphDBConnection

# ============== 配置 ==============
MIN_DOCS = 2
MAX_DOCS = 10
# ==================================


def estimate_candidate_pairs():
    settings = Settings()
    
    print(f"连接到 Neo4j: {settings.neo4j_uri}")
    graph_conn = GraphDBConnection(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )
    
    try:
        # 查询：按实体名称分组，统计每个名称出现在多少个文档中
        query = """
        MATCH (e:Entity)
        WHERE e.is_named_entity = true
        WITH e.name as name, count(e) as count
        RETURN count as doc_count, count(*) as entity_name_count
        ORDER BY doc_count
        """
        
        print("\n=== 实体名称分布统计 ===")
        print(f"{'出现文档数':<12} {'实体名称数':<12} {'候选对数':<12}")
        print("-" * 40)
        
        total_pairs_by_range = defaultdict(int)
        distribution = {}
        
        with graph_conn._driver.session() as session:
            results = session.run(query)
            for record in results:
                doc_count = record["doc_count"]
                entity_name_count = record["entity_name_count"]
                # 每个名称组内的候选对数 = C(doc_count, 2) * entity_name_count
                pairs_per_name = doc_count * (doc_count - 1) // 2
                total_pairs = pairs_per_name * entity_name_count
                distribution[doc_count] = (entity_name_count, total_pairs)
                print(f"{doc_count:<12} {entity_name_count:<12} {total_pairs:<12}")
        
        print("-" * 40)
        
        # 计算指定范围内的候选对
        total_names = 0
        total_pairs = 0
        for doc_count, (name_count, pairs) in distribution.items():
            if MIN_DOCS <= doc_count <= MAX_DOCS:
                total_names += name_count
                total_pairs += pairs
        
        print(f"\n=== 当前配置: MIN_DOCS={MIN_DOCS}, MAX_DOCS={MAX_DOCS} ===")
        print(f"符合条件的实体名称数: {total_names}")
        print(f"将生成的候选对数量: {total_pairs}")
        
        # 估算 LLM 调用成本
        print(f"\n=== 成本估算 ===")
        print(f"LLM 调用次数: {total_pairs}")
        avg_time_per_call = 2  # 假设每次 LLM 调用 2 秒
        print(f"预计耗时 (单线程): {total_pairs * avg_time_per_call / 60:.1f} 分钟")
        print(f"预计耗时 (8线程): {total_pairs * avg_time_per_call / 60 / 8:.1f} 分钟")
        
    finally:
        graph_conn.close()


if __name__ == "__main__":
    estimate_candidate_pairs()
