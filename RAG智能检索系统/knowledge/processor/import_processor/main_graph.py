# 内置包
import json

# 三方包
from langgraph.graph.state import StateGraph,CompiledStateGraph
from langgraph.graph import  START,END
from IPython.display import display
# 项目包
from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.nodes.md_to_img_node import MarkdownToImgNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode

from knowledge.processor.import_processor.state import ImportGraphState

def import_router(state:ImportGraphState):
    """
    根据用户上传文件的类型，判断state中is_pdf_read_enabled is_md_read_enabled绝定下一步节点
    Args:
        state: ImportGraphState
    Returns:ImportGraphState
    """
    if state.get('is_pdf_read_enabled'):
        return 'pdf_to_md_node'
    elif state.get('is_md_read_enabled'):
        return  'md_to_img_node'
    return END

def builder_graph()->CompiledStateGraph:
    """
    主要职责：
    1、定义运行时图节点状态
    2、定义条件边、普通边
    3、定义节点（入口节点、业务节点）
    4、创建图，返回运行时的状态
    Returns:

    """
    work_flow = StateGraph(state_schema=ImportGraphState)  # type:ignore

    work_flow.set_entry_point('entry_node')

    node_name_obj = {
        'pdf_to_md_node':PdfToMdNode(),
        'md_to_img_node':MarkdownToImgNode(),
        'entry_node':EntryNode(),
        'document_split_node':DocumentSplitNode (),
        'milvus_import_node': MilvusImportNode(),
        'embedding_chunks_node': EmbeddingChunksNode(),
        'item_name_recognition_node': ItemNameRecognitionNode(),

    }

    for node_name,node_obj in node_name_obj.items():
        work_flow.add_node(node_name,node_obj)

    path_map = {
        'pdf_to_md_node':'pdf_to_md_node',
        'md_to_img_node':'md_to_img_node',
        END:END
            }
    work_flow.add_conditional_edges('entry_node',import_router,path_map=path_map)
    work_flow.add_edge('pdf_to_md_node','md_to_img_node')
    work_flow.add_edge('md_to_img_node', 'document_split_node')
    work_flow.add_edge('document_split_node', 'item_name_recognition_node')
    work_flow.add_edge('item_name_recognition_node', 'embedding_chunks_node')
    work_flow.add_edge('embedding_chunks_node', 'milvus_import_node')
    work_flow.add_edge('milvus_import_node', END)

    compiled_state_graph = work_flow.compile()

    display(compiled_state_graph)

    return compiled_state_graph


builder_graph = builder_graph()



def run_graph():
    graph_state = {
        "import_file_path":r"G:\AI-project\shopkeeper_brain\knowledge\test\万用表RS-12的使用.pdf",
        "file_dir":r"G:\AI-project\shopkeeper_brain\knowledge\test\万用表RS-12的使用"
    }

    for event in builder_graph.stream(graph_state):
        final_state = {}
        for node_name,node_state in event.items():
            print(f"当前执行节点：{node_name}")
            print(f"当前节点状态：{node_state}")
            final_state = node_state
    return final_state

if __name__ =="__main__":
    setup_logging()
    final_state = run_graph()
    print(json.dumps(final_state,ensure_ascii=False,indent=4))










