import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

class Settings:
    """
    一个用于加载和提供项目配置的单例类。
    它会从 .env 文件和 configs/config.yaml 文件中加载配置。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_configs()
        return cls._instance

    def _load_configs(self):
        # 获取项目根目录
        # 由于此文件在 bridgerag 包内，我们需要获取其父目录的父目录
        self.project_root = Path(__file__).resolve().parent.parent
        self._load_env()
        self._load_yaml_config()

    def _load_env(self):
        """Loads environment variables from .env file."""
        # 加载 .env 文件
        dotenv_path = self.project_root / ".env"
        load_dotenv(dotenv_path=dotenv_path)

        # --- 从 .env 文件加载环境变量 ---
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")

        self.milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        self.siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY", "sk-kevnxgxmsbrzhvqihierepcvnaucpvlxuktfchotrojchuyg")

        self.celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        self.celery_result_backend_url = os.getenv("CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

    def _load_yaml_config(self):
        """Loads YAML configuration from config.yaml."""
        # 加载 YAML 配置文件
        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._yaml_data = yaml.safe_load(f)

        # --- 从 config.yaml 文件加载配置 ---
        # LLM 配置
        llm_config = self._yaml_data.get("llm", {})
        self.llm_api_type = llm_config.get("llm_api_type")
        self.embedding_api_type = llm_config.get("embedding_api_type", "local")

        # vLLM
        self.vllm_generation_model_name = llm_config.get("vllm_generation_model_name")
        self.vllm_tokenizer_path = llm_config.get("vllm_tokenizer_path")
        self.vllm_embedding_model_name = llm_config.get("vllm_embedding_model_name", "nomic-embed-text-v1.5")
        self.embedding_dim = llm_config.get("embedding_dim", 768) # 正确的位置
        self.vllm_host = llm_config.get("vllm_host")
        self.vllm_port = llm_config.get("vllm_port")
        self.vllm_embedding_host = llm_config.get("vllm_embedding_host", "localhost")
        self.vllm_embedding_port = llm_config.get("vllm_embedding_port", 8990)

        # SiliconFlow
        self.siliconflow_api_base = llm_config.get("siliconflow_api_base")
        self.siliconflow_generation_model_name = llm_config.get(
            "siliconflow_generation_model_name"
        )
        self.siliconflow_embedding_model_name = llm_config.get(
            "siliconflow_embedding_model_name"
        )

        # Local Embedding
        self.local_embedding_model_path = llm_config.get("local_embedding_model_path")

        # 数据库配置
        db_config = self._yaml_data.get("database", {})
        self.chunk_collection_name = db_config.get("chunk_collection_name")
        self.entity_collection_name = db_config.get("entity_collection_name")
        self.summary_collection_name = db_config.get("summary_collection_name")
        self.minio_bucket_name = db_config.get("minio_bucket_name")

        # 数据处理配置
        processing_config = self._yaml_data.get("processing", {})
        self.embedding_dimension = processing_config.get("embedding_dimension")
        self.text_chunk_size = processing_config.get("text_chunk_size", 1024)
        self.overlap_token_size = processing_config.get("overlap_token_size", 64)
        self.force_rewrite = processing_config.get("force_rewrite", False)
        self.max_workers = processing_config.get("max_workers", 16)

        # 数据库 Schema 配置 (从新的 'database_schema' 块加载)
        db_schema_config = self._yaml_data.get("database_schema", {})
        self.chunk_max_length = db_schema_config.get("chunk_max_length", 16384)
        self.entity_summary_max_length = db_schema_config.get("entity_summary_max_length", 4096)
        self.summary_max_length = db_schema_config.get("summary_max_length", 8192)

        # 检索服务配置
        retrieval_config = self._yaml_data.get("retrieval", {})
        self.RERANK_TOP_K = retrieval_config.get("rerank_top_k", 5)

        # 实体扩展配置
        entity_expansion_config = self._yaml_data.get("entity_expansion", {})
        self.SAME_AS_ENTITY_EXPANSION_LIMIT = entity_expansion_config.get("same_as_entity_expansion_limit", 5)
        self.CHUNK_RETRIEVAL_LIMIT = entity_expansion_config.get("chunk_retrieval_limit", 2)

        # 合成服务配置
        synthesis_config = self._yaml_data.get("synthesis", {})
        self.MAX_SYNTHESIS_ATTEMPTS = synthesis_config.get("max_synthesis_attempts", 3)

        # 日志配置
        logging_config = self._yaml_data.get("logging", {})
        self.failed_chunks_dir = self.project_root / logging_config.get("failed_chunks_dir", "logs/failed_chunks")

        # 数据路径配置
        data_config = self._yaml_data.get("data", {})
        self.raw_data_path = self.project_root / data_config.get("raw_data_path", "data.jsonl")


# 创建一个全局唯一的配置实例，项目中其他地方都将导入这个实例
settings = Settings()

# 为了方便调试，可以在这里打印加载的配置
if __name__ == "__main__":
    print("--- Loaded Settings ---")
    print(f"Project Root: {settings.project_root}")
    print(f"LLM API Type: {settings.llm_api_type}")
    print(f"Embedding API Type: {settings.embedding_api_type}")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"MinIO Endpoint: {settings.minio_endpoint}")
    print(f"Milvus Entity Collection: {settings.entity_collection_name}")
    print(f"Celery Broker URL: {settings.celery_broker_url}")
    print(f"Local Embedding Path: {settings.local_embedding_model_path}")
    print("-----------------------")