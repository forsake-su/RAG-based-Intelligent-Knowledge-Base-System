from functools import lru_cache
# from knowledge.services.import_file_service import UploadFileService
# from knowledge.services.task_service import TaskService
from knowledge.services.query_service import QueryService
from knowledge.services.upload_service import UploadService

# @lru_cache
# def get_task_service() -> TaskService:
#     return TaskService()

@lru_cache  # 缓存注解，将实例对象缓存一份，当缓存数量达到阈值，淘汰过时数据
def get_upload_file_service() -> UploadService:
    return UploadService()

@lru_cache
def get_query_service() -> QueryService:
    return QueryService()