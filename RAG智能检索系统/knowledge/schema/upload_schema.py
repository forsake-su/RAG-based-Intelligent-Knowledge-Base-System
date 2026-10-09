from typing import Dict,List
from pydantic import BaseModel,Field

class UploadResponse(BaseModel):
    """文件上传响应 —— POST /upload 返回"""
    message:str = Field(...,description='响应信息')  # = Field(...) 表示 该字段没有默认值，因此是必填的
    task_id:str = Field(...,description='任务ID')

class TaskStatusResponse(BaseModel):
    """任务状态响应 —— GET /status/{task_id} 返回"""
    status:str = Field(...,description='任务状态')
    done_list : List[str] = Field(...,description='已完成节点列表')
    running_list:List[str] = Field(...,description='正在运行节点列表')
    durations:Dict[str,float] = Field(default=dict,description='各节点耗时（秒）')

