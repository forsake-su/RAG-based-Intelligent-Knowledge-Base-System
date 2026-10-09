import logging,secrets

from typing import List,Dict,Any

from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.task_util import update_task_status,get_task_status,get_task_result,TASK_STATUS_FAILED,TASK_STATUS_COMPLETED,TASK_STATUS_PROCESSING

logger = logging.getLogger(__name__)

class QueryService:

    def generate_session_id(self)->str:
        return secrets.token_hex(4)

    def generate_task_id(self)->str:
        return secrets.token_hex(3)

    def  run_query_graph(self,task_id:str,session_id:str,original_query:str,is_stream:bool):
        """
        执行查询流程
        Args:
            task_id:
            session_id:
            original_query:
            is_stream:

        Returns:

        """
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        try:
            default_state = {
                'original_query':original_query,
                'session_id':session_id,
                'task_id':task_id,
                'is_stream':is_stream
            }
            query_app.invoke(default_state)

            update_task_status(task_id,TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"查询流程执行失败，原因：{str(e)}",exc_info=True)
            update_task_status(task_id, TASK_STATUS_FAILED)

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")