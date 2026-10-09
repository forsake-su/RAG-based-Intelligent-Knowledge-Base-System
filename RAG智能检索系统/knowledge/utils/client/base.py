import os
import logging,threading
from typing import TypeVar,Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseClientManager:
    """
    客户端管理器基类，提供：
    - _require_env()：环境变量校验
    - _get_or_create()：双重检查锁模板方法

    子类只需要关注「怎么创建客户端」，不用重复写锁逻辑。
    """

    @staticmethod
    def _require_env(key:str)->str:
        """
        读取环境变量，缺失时抛异常
        Args:
            key: 要读取的环境变量名
        Returns:环境变量值
        """
        value = os.getenv(key)

        if not value:
            raise EnvironmentError(f'缺少必须的环境变量：{key}')
        return value

    @classmethod
    def _get_or_create(cls,attr_name:str,lock:threading.Lock,factory):
        """
        双重检查锁，获取实例
        Args:
            attr_name: 类属性名
            lock: 线程锁
            factory: 工厂函数，返回客户端实例
        Returns:加载或创建客户端实例
        """
        instance = getattr(cls,attr_name,None)
        if instance is not None:
            return instance

        with lock:
            instance = getattr(cls, attr_name, None)
            if instance is not None:
                return instance
            #调用工厂函数，获取实例
            instance =factory()
            setattr(cls,attr_name,instance)
            return instance