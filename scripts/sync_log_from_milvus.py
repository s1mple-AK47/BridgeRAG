"""
从 Milvus 同步已处理文档列表到日志文件。
用于修复日志和数据库不一致的情况。
"""
import sys
from pathlib import Path
from urllib.parse import urlparse

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import settings
from bridgerag.database.vector_db import VectorDBConnection
from pymilvus import Collection

def main():
    # 连接 Milvus
    parsed_uri = urlparse(settings.milvus_uri)
    VectorDBConnection(host=parsed_uri.hostname, port=parsed_uri.port, alias="default")
    
    # 从 summary 集合获取所有 doc_id（每个文档只有一条摘要记录）
    collection_name = settings.summary_collection_name
    collection = Collection(collection_name, using="default")
    collection.load()
    
    # 分页查询所有 doc_id
    all_doc_ids = set()
    batch_size = 10000
    offset = 0
    
    while True:
        results = collection.query(
            expr="doc_id != ''",
            output_fields=["doc_id"],
            limit=batch_size,
            offset=offset
        )
        
        if not results:
            break
            
        for r in results:
            all_doc_ids.add(r["doc_id"])
        
        if len(results) < batch_size:
            break
            
        offset += batch_size
    
    processed_ids = sorted(all_doc_ids)
    
    # 写入日志文件
    log_path = project_root / 'logs' / 'successful_documents.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        for doc_id in processed_ids:
            f.write(f'{doc_id}\n')
    
    print(f'已从 Milvus 同步 {len(processed_ids)} 篇文档到日志文件')

if __name__ == "__main__":
    main()
