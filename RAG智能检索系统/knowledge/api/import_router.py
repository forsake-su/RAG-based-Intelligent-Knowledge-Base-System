import os.path
import uvicorn

from fastapi import FastAPI,UploadFile,Depends,BackgroundTasks,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse,TaskStatusResponse
from knowledge.services.upload_service import UploadService
from knowledge.core.deps import get_upload_file_service
from knowledge.utils.task_util import get_task_info

def create_app():
    """
    创建FastAPI实例
    Returns:

    """
    app = FastAPI(description="掌柜智库",version='v1.0')
    app.add_middleware(
        CORSMiddleware,
        allow_origins = ['*'],
        allow_credentials = False,
        allow_methods = ['*'],
        allow_headers = ['*'],
    )

    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        app.mount('/front',StaticFiles(directory=page_dir))

    register_router(app)
    return  app

def register_router(app:FastAPI):

    @app.get('import.html')
    def import_page():
        path = get_front_page_dir()/'import'
        if not os.path.exists(path):
            raise HTTPException(status_code=404,detail='import.html not found')
        return FileResponse(path)

    # 上传请求
    @app.post('/upload',response_model=UploadResponse)
    def upload_endpoint(file:UploadFile,background_tasks:BackgroundTasks,upload_service:UploadService=Depends(get_upload_file_service)):
        """
        处理文件上传
        Args:
            file:
            background_tasks:
            upload_service:

        Returns:

        """
        task_id,import_file_path,file_dir = upload_service.process_upload_file(file)
        background_tasks.add_task(upload_service.run_import_graph,task_id,import_file_path,file_dir)

        return UploadResponse(message=f"{file.filename}上传成功",task_id=task_id)

    @app.get('/status/{task_id}')
    def get_task_status_endpoint(task_id:str):
        """
        查询任务节点状态
        Args:
            task_id:

        Returns:

        """
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)
if __name__ == '__main__':
    # param1:fastapi实例
    # param2:启动的服务器地址
    # param3:启动的服务端口
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8000, log_level="info")