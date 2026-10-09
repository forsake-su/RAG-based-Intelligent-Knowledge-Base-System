import os,asyncio,uvicorn

from fastapi import FastAPI,Depends,Request,BackgroundTasks,HTTPException
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.paths import get_front_page_dir
from knowledge.core.deps import get_query_service
from knowledge.schema.query_schema import QueryRequest,QueryResponse,StreamSubmitResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import sse_generator,create_sse_queue


def create_app():

    app = FastAPI(title='Query Service',description='知识库查询服务')

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    register_routes(app)
    return app

def register_routes(app: FastAPI):
    """

    Args:
        app:

    Returns:

    """
    @app.get('/chat.html')
    async  def chat_page():
        path = get_front_page_dir()/'chat.html'
        if not os.path.exists(path):
            raise HTTPException(status_code=404,detail='chat.html not found')
        return FileResponse(path)

    @app.post('/query')
    async def query(
            request:QueryRequest,
            background_tasks:BackgroundTasks,
            service:QueryService =Depends()
    ):

        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        # 流式输出
        if request.is_stream:
            create_sse_queue(task_id)
            background_tasks.add_task(service.run_query_graph,
                                      task_id=task_id,
                                      session_id=session_id,
                                      original_query=request.query,
                                      is_stream = True)

            return StreamSubmitResponse(message='查询请求已提交',session_id=session_id,task_id=task_id)

        # 非流式输出
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None,service.run_query_graph,task_id,session_id,request.query,False)
        answer = service.get_answer(task_id)

        return QueryResponse(message="处理完成", session_id=session_id, answer=answer)






if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8001)