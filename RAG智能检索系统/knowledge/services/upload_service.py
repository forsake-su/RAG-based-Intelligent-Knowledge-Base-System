import logging,shutil,time,secrets
import os
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.main_graph import builder_graph
from knowledge.utils.task_util import update_task_status,add_running_task,add_done_task,add_node_duration,TASK_STATUS_FAILED,TASK_STATUS_COMPLETED,TASK_STATUS_PROCESSING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadService:
    """处理文件上传流程"""
    def get_base_dir(self)->str:
        time_now = datetime.now().strftime("%Y%m%d")
        return get_local_base_dir()/time_now

    def run_import_graph(self,task_id:str,import_file_path:str,file_dir:str):
        """
        运行图谱流程
        Args:
            task_id:
            import_file_path:
            file_dir:

        Returns:

        """
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        graph_state = {
            'task_id':task_id,
            'import_file_path':import_file_path,
            'file_dir':file_dir
        }

        try:
            for event in builder_graph.stream(graph_state):
                for key,value in event.items():
                    logger.info(f"当前执行节点-->{key}")
            update_task_status(task_id,TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"【{task_id}】执行导入过程异常，原因：{str(e)}")
            update_task_status(task_id,TASK_STATUS_FAILED)

    def process_upload_file(self,file:UploadFile):
        """
        处理文件上传
        1、本地临时目录
        2、远程minio
        Args:
            file: 客户端导入文件

        Returns:

        """
        task_id = secrets.token_hex(4) # 生成4个字节，即8位字符的随机数
        add_running_task(task_id,'upload_file')
        start_time = time.time()

        base_file_dir = self.get_base_dir()
        file_dir = base_file_dir/task_id
        import_file_path = self.save_upload_file_to_local(file,file_dir)
        self.save_upload_file_to_minio(import_file_path,file.filename)
        end_time = time.time()
        add_done_task(task_id,'upload_file')
        add_node_duration(task_id,'upload_file',duration=end_time-start_time)
        return task_id,import_file_path,file_dir

    def save_upload_file_to_local(self, file:UploadFile, file_dir:str):
        """
        保存文件到临时目录
        Args:
            file: 上传文件对象
            file_dir: 上传文件所在目录

        Returns:

        """
        os.makedirs(file_dir,exist_ok=True)
        import_file_path = file_dir/file.filename
        try:
            with open(import_file_path,'wb') as f:
                shutil.copyfileobj(file.file,f)  # f.write不适用写入大文件，写入时占用大量内存。shutil可以按字节大小逐步写入，占用内存小，适用于处理大文件
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败，原因：{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败 原因:{str(e)}")

        return import_file_path
    def save_upload_file_to_minio(self, import_file_path:str, filename:str):
        """上传文件至minio"""

        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"minio客户端调用失败，原因：{str(e)}")
            return

        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        try:
            minio_client.fput_object(bucket_name,object_name,import_file_path)
        except Exception as e:
            logger.error(f"{filename}上传至Minio失败，原因：{str(e)}")













