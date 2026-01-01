from celery import Celery
from bridgerag.config import settings

# 创建Celery应用实例
# 我们使用'bridgerag'作为主模块名称
celery_app = Celery(
    "bridgerag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend_url,
    include=["bridgerag.offline.tasks"],  # 自动发现任务的模块路径
)

# 可选的Celery配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    # 增加任务路由和队列设置，为未来的扩展做准备
    task_routes={
        "offline.*": {"queue": "offline_processing"},
    },
)

if __name__ == "__main__":
    # 这个入口点主要用于测试或直接从命令行启动worker
    # 例如: celery -A bridgerag.celery_app worker --loglevel=info
    celery_app.start()
