from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import logging

from bridgerag.online.main import OnlineQueryProcessor
from bridgerag.online.schemas import SynthesisDecision, QueryRequest, QueryResponse
from bridgerag.utils.logging_config import configure_logging

# 配置日志
configure_logging()
logger = logging.getLogger(__name__)

# 全局变量，用于在应用的生命周期内持有 OnlineQueryProcessor 实例
processor_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期管理器。
    在应用启动时初始化资源，在应用关闭时释放资源。
    """
    global processor_instance
    logger.info("应用启动中，正在初始化 OnlineQueryProcessor...")
    try:
        processor_instance = OnlineQueryProcessor()
        logger.info("OnlineQueryProcessor 初始化成功。")
        yield
    finally:
        logger.info("应用关闭中，正在清理资源...")
        if processor_instance and hasattr(processor_instance, 'close'):
            processor_instance.close()
        logger.info("资源清理完成。")


app = FastAPI(
    title="BridgeRAG API",
    description="一个用于与 BridgeRAG 系统交互的 API。",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["General"])
async def read_root():
    """
    根路径，用于检查服务是否正在运行。
    """
    return {"message": "欢迎来到 BridgeRAG API！"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    接收用户问题并返回经过 BridgeRAG 处理后的答案。
    """
    if not processor_instance:
        raise HTTPException(status_code=503, detail="查询处理器尚未初始化。")
    
    logger.info(f"收到查询请求: {request.question}")
    
    try:
        response = await processor_instance.process_query(
            question=request.question,
            max_turns=request.max_turns
        )
        return response
    except Exception as e:
        logger.critical(f"处理查询 '{request.question}' 时发生严重错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="处理请求时发生内部服务器错误。")


if __name__ == "__main__":
    # 使用 uvicorn 启动应用
    # 命令行运行: uvicorn bridgerag.online.api_server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
