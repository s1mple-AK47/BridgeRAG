import logging
from celery import chain, group
from typing import Dict, Any, List, Tuple

# 导入Celery任务
from .tasks import (
    process_document_task,
    save_knowledge_graph_task,
)

logger = logging.getLogger(__name__)


def trigger_batch_processing(batch: List[Tuple[str, Dict[str, Any]]]):
    """
    Triggers the Celery processing workflow for a batch of documents.
    """
    logger = logging.getLogger(__name__)

    if not batch:
        logger.warning("Received an empty batch. Nothing to process.")
        return None

    # Map Phase: Process and save each document individually.
    # The chain ensures that saving only happens after successful processing.
    map_tasks = group(
        chain(
            process_document_task.s(doc_id=doc_id, document_data=doc_data), 
            save_knowledge_graph_task.s()
        ) 
        for doc_id, doc_data in batch
    )

    # The workflow now only consists of the map phase.
    # Entity linking has been removed and will be run manually.
    workflow = map_tasks
    
    try:
        result = workflow.apply_async()
        logger.info(f"Successfully submitted batch workflow to Celery. Group ID: {result.id}")
        return result
    except Exception as e:
        logger.error(f"Failed to submit batch workflow to Celery: {e}", exc_info=True)
        return None

