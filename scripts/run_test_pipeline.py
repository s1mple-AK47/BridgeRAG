"""
端到端测试脚本：处理 HotpotQA 数据集的前 20 条数据，走完整个离线+在线流程。

使用方法：
1. 确保 Docker 服务已启动: docker-compose up -d
2. 确保 vLLM 服务已启动
3. 在一个终端启动 Celery Worker:
   celery -A bridgerag.celery_app worker --loglevel=info -c 4 -Q offline_processing
4. 在另一个终端运行此脚本:
   python scripts/run_test_pipeline.py
"""

import json
import logging
import sys
import time
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.utils.logging_config import setup_logging
from bridgerag.config import settings

setup_logging()
logger = logging.getLogger(__name__)

# ============== 配置 ==============
TEST_DOC_LIMIT = 20  # 测试文档数量
TEST_QUESTION_LIMIT = 5  # 测试问题数量
# ==================================


def step1_convert_data():
    """步骤1：转换数据格式（只取前 N 条）"""
    logger.info("=" * 60)
    logger.info("步骤 1: 转换数据格式")
    logger.info("=" * 60)
    
    wiki_input = project_root / "datas" / "wiki_pages_final.jsonl"
    qa_input = project_root / "datas" / "hotpot_qa_filtered.jsonl"
    
    docs_output = project_root / "test_docs.jsonl"
    questions_output = project_root / "test_questions.jsonl"
    
    if not wiki_input.exists():
        logger.error(f"文档文件不存在: {wiki_input}")
        return False
    
    if not qa_input.exists():
        logger.error(f"问答文件不存在: {qa_input}")
        return False
    
    # 转换文档（取前 N 条）
    doc_count = 0
    seen_ids = set()
    with open(wiki_input, 'r', encoding='utf-8') as infile, \
         open(docs_output, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if doc_count >= TEST_DOC_LIMIT:
                break
            data = json.loads(line.strip())
            doc_id = data.get("title", "").strip()
            content = data.get("content", "").strip()
            if doc_id and content and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                outfile.write(json.dumps({"id": doc_id, "text": content}, ensure_ascii=False) + '\n')
                doc_count += 1
    
    logger.info(f"已转换 {doc_count} 篇文档 -> {docs_output}")
    
    # 转换问答（取前 N 条）
    question_count = 0
    with open(qa_input, 'r', encoding='utf-8') as infile, \
         open(questions_output, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, 1):
            if question_count >= TEST_QUESTION_LIMIT:
                break
            data = json.loads(line.strip())
            question = data.get("question", "").strip()
            answer = data.get("answer", "").strip()
            titles = data.get("title", [])
            if question and answer:
                unique_titles = list(dict.fromkeys(titles))
                outfile.write(json.dumps({
                    "id": f"q_{i}",
                    "question": question,
                    "answer": answer,
                    "ids": unique_titles
                }, ensure_ascii=False) + '\n')
                question_count += 1
    
    logger.info(f"已转换 {question_count} 个问题 -> {questions_output}")
    return True


def step2_initialize_databases():
    """步骤2：初始化数据库"""
    logger.info("=" * 60)
    logger.info("步骤 2: 初始化数据库")
    logger.info("=" * 60)
    
    from urllib.parse import urlparse
    from bridgerag.database.vector_db import VectorDBConnection
    from bridgerag.database import vector_ops
    
    try:
        parsed_milvus_uri = urlparse(settings.milvus_uri)
        vector_db_conn = VectorDBConnection(
            host=parsed_milvus_uri.hostname,
            port=parsed_milvus_uri.port,
            alias="default"
        )
        
        # 创建集合
        vector_ops.create_chunk_collection(
            collection_name=settings.chunk_collection_name,
            dense_dim=settings.embedding_dim,
            text_max_length=settings.chunk_max_length
        )
        vector_ops.create_entity_collection(
            collection_name=settings.entity_collection_name,
            dense_dim=settings.embedding_dim,
            text_max_length=settings.entity_summary_max_length
        )
        vector_ops.create_summary_collection(
            collection_name=settings.summary_collection_name,
            dense_dim=settings.embedding_dim,
            text_max_length=settings.summary_max_length
        )
        
        vector_db_conn.close()
        logger.info("Milvus 集合初始化完成")
        return True
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}", exc_info=True)
        return False


def step3_run_offline_pipeline():
    """步骤3：运行离线流水线（通过 Celery）"""
    logger.info("=" * 60)
    logger.info("步骤 3: 运行离线流水线")
    logger.info("=" * 60)
    
    from bridgerag.offline.pipeline import trigger_batch_processing
    from bridgerag.database.object_storage import ObjectStorageConnection
    import bridgerag.database.object_storage_ops as storage_ops
    
    data_file = project_root / "test_docs.jsonl"
    
    if not data_file.exists():
        logger.error(f"测试数据文件不存在: {data_file}")
        return False
    
    # 获取已处理的文档
    processed_ids = set()
    try:
        minio_conn = ObjectStorageConnection(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key
        )
        if minio_conn.client.bucket_exists(settings.minio_bucket_name):
            object_names = storage_ops.list_objects(minio_conn.client, settings.minio_bucket_name)
            processed_ids = {Path(obj_name).stem for obj_name in object_names}
        logger.info(f"已处理文档数: {len(processed_ids)}")
    except Exception as e:
        logger.warning(f"检查 MinIO 时出错: {e}")
    
    # 加载文档
    documents = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data["id"] not in processed_ids:
                documents.append(data)
    
    if not documents:
        logger.info("所有文档已处理完毕")
        return True
    
    logger.info(f"待处理文档数: {len(documents)}")
    
    # 提交任务
    batch_for_pipeline = [(doc["id"], {"text": doc["text"]}) for doc in documents]
    
    logger.info("正在提交任务到 Celery...")
    task_group_result = trigger_batch_processing(batch_for_pipeline)
    
    if task_group_result:
        logger.info(f"任务已提交，任务组 ID: {task_group_result.id}")
        logger.info("等待任务完成（超时时间: 30 分钟）...")
        try:
            task_group_result.get(timeout=1800)  # 30 分钟超时
            logger.info("所有任务已完成！")
            return True
        except Exception as e:
            logger.error(f"任务执行出错: {e}")
            return False
    else:
        logger.error("任务提交失败")
        return False


def step4_run_entity_linking():
    """步骤4：运行实体链接"""
    logger.info("=" * 60)
    logger.info("步骤 4: 运行实体链接")
    logger.info("=" * 60)
    
    from urllib.parse import urlparse
    from bridgerag.database.graph_db import GraphDBConnection
    from bridgerag.database.vector_db import VectorDBConnection
    from bridgerag.core.llm_client import LLMClient
    from bridgerag.offline.steps.link_entities import run_entity_linking
    
    try:
        llm_client = LLMClient()
        gdb_conn = GraphDBConnection(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        parsed_milvus_uri = urlparse(settings.milvus_uri)
        VectorDBConnection(host=parsed_milvus_uri.hostname, port=parsed_milvus_uri.port)
        
        run_entity_linking(
            driver=gdb_conn._driver,
            llm_client=llm_client,
            milvus_collection_name=settings.entity_collection_name,
            max_workers=4
        )
        
        gdb_conn.close()
        logger.info("实体链接完成")
        return True
    except Exception as e:
        logger.error(f"实体链接失败: {e}", exc_info=True)
        return False


def step5_run_online_query():
    """步骤5：运行在线查询测试"""
    logger.info("=" * 60)
    logger.info("步骤 5: 运行在线查询测试")
    logger.info("=" * 60)
    
    from bridgerag.online.main import OnlineQueryProcessor
    
    questions_file = project_root / "test_questions.jsonl"
    
    if not questions_file.exists():
        logger.error(f"测试问题文件不存在: {questions_file}")
        return False
    
    # 加载问题
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    logger.info(f"加载了 {len(questions)} 个测试问题")
    
    try:
        processor = OnlineQueryProcessor()
        
        results = []
        for i, q in enumerate(questions, 1):
            logger.info(f"\n--- 问题 {i}/{len(questions)} ---")
            logger.info(f"问题: {q['question']}")
            logger.info(f"标准答案: {q['answer']}")
            
            start_time = time.time()
            result = processor.process_query(q['question'])
            duration = time.time() - start_time
            
            logger.info(f"LLM 答案: {result.answer}")
            logger.info(f"相关文档: {result.main_documents}")
            logger.info(f"耗时: {duration:.2f}s")
            
            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": q["answer"],
                "LLM_answer": result.answer,
                "LLM_docs": result.main_documents,
                "duration": round(duration, 2)
            })
        
        processor.close()
        
        # 保存结果
        output_file = project_root / "test_results.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        logger.info(f"\n结果已保存到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"在线查询失败: {e}", exc_info=True)
        return False


def main():
    """主函数：按顺序执行所有步骤"""
    logger.info("=" * 60)
    logger.info("BridgeRAG 端到端测试")
    logger.info(f"测试文档数: {TEST_DOC_LIMIT}, 测试问题数: {TEST_QUESTION_LIMIT}")
    logger.info("=" * 60)
    
    steps = [
        ("转换数据", step1_convert_data),
        ("初始化数据库", step2_initialize_databases),
        ("离线流水线", step3_run_offline_pipeline),
        ("实体链接", step4_run_entity_linking),
        ("在线查询", step5_run_online_query),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n>>> 开始执行: {step_name}")
        success = step_func()
        if not success:
            logger.error(f"<<< {step_name} 失败，测试中止")
            return
        logger.info(f"<<< {step_name} 完成")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 端到端测试全部完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
