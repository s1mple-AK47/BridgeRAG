from neo4j import GraphDatabase, Driver
import logging

# 在数据库模块中使用日志记录是一个好习惯
logger = logging.getLogger(__name__)

class GraphDBConnection:
    """
    管理与 Neo4j 图数据库的连接。
    这个类被设计为 Neo4j 驱动程序 (driver) 的单例包装器。
    """
    _driver: Driver | None = None

    def __init__(self, uri, user, password):
        """
        初始化连接。这个方法应该在应用程序启动时只调用一次。
        
        参数:
            uri (str): Neo4j 数据库的 URI (例如, "neo4j://localhost:7687")。
            user (str): 用于认证的用户名。
            password (str): 用于认证的密码。
        """
        if self.__class__._driver is None:
            try:
                logger.info(f"正在为 URI 初始化 Neo4j 驱动: {uri}")
                self.__class__._driver = GraphDatabase.driver(uri, auth=(user, password))
                self._driver.verify_connectivity()
                logger.info("Neo4j 驱动初始化并成功验证连接。")
            except Exception as e:
                logger.error(f"创建或验证 Neo4j 驱动失败: {e}")
                self.__class__._driver = None # 失败时确保驱动为 None
                raise
        else:
            logger.warning("Neo4j 驱动已经被初始化。")

    def close(self):
        """关闭与 Neo4j 数据库的连接。"""
        if self._driver and self._driver._closed is False:
            self._driver.close()
            logger.info("Neo4j 驱动连接已关闭。")

    def get_driver(self) -> Driver:
        """返回底层的 Neo4j 驱动实例。"""
        return self._driver 