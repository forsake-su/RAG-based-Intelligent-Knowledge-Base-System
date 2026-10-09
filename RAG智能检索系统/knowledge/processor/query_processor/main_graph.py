import json

from langgraph.graph.state import StateGraph,CompiledStateGraph
from langgraph.graph import START,END,StateGraph
from dotenv import load_dotenv

from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.nodes.item_name_confirmed_node import ItemNameConfirmedNode
from knowledge.processor.query_processor.nodes.hybrid_vector_search_node import HybridVectorSearchNode
from knowledge.processor.query_processor.nodes.hyde_vector_search_node import HydeVectorSearchNode
from knowledge.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode
from knowledge.processor.query_processor.nodes.rrf_node import RrfNode
from knowledge.processor.query_processor.nodes.rerank_node import RerankNode
from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutputNode

load_dotenv()


def route_after_item_confirm(state:QueryGraphState)->bool:
    """
    商品名称确认后的路由逻辑。
    Args:
        state:

    Returns:

    """
    if state.get('answer'):
        return True
    return False

def create_query_graph()->CompiledStateGraph:
    """
    创建查询流程图。
    Returns:编译后的 StateGraph 实例。

    """
    workflow = StateGraph(QueryGraphState)   # type:ignore

    nodes = {
        'item_name_confirmed_node':ItemNameConfirmedNode(),
        "multi_search": lambda x: x,  # 虚拟节点
        'hybrid_vector_search_node':HybridVectorSearchNode(),
        "hyde_vector_search_node": HydeVectorSearchNode(),
        "web_mcp_search_node": WebMcpSearchNode(),
        "join": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf_node": RrfNode(),
        "reranker_node": RerankNode(),
        "answer_output_node": AnswerOutputNode(),
    }

    for name,node in nodes.items():
        workflow.add_node(name,node) # type:ignore

    workflow.set_entry_point('item_name_confirmed_node')
    path_map = {False:'multi_search',
                True:'answer_output_node'}
    workflow.add_conditional_edges(
        'item_name_confirmed_node',route_after_item_confirm,path_map=path_map
    )

    workflow.add_edge("multi_search", "hybrid_vector_search_node")
    workflow.add_edge("multi_search", "hyde_vector_search_node")
    workflow.add_edge("multi_search", "web_mcp_search_node")

    workflow.add_edge("hybrid_vector_search_node", "join")
    workflow.add_edge("hyde_vector_search_node", "join")
    workflow.add_edge("web_mcp_search_node", "join")

    workflow.add_edge("join", "rrf_node")
    workflow.add_edge("rrf_node", "reranker_node")
    workflow.add_edge("reranker_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


query_app = create_query_graph()

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 查询流程主图 (main_graph)")
    print("=" * 60)

    # ---- 测试场景 1：商品名明确，走完整 pipeline ----
    print("\n【场景 1】: 商品名明确，走完整 pipeline")
    print("-" * 60)

    mock_state_1 = {
        "original_query": "RS-12 数字万用表如何测量直流电压？",
        "session_id": "test_session_main_graph",
        "task_id": "test_task_001",
        "is_stream": False,
    }

    result_1 = query_app.invoke(mock_state_1)


    print(f"\n  【结果】:")
    print(f"  商品名: {result_1.get('item_names')}")
    print(f"  重写查询: {result_1.get('rewritten_query')}")
    answer_1 = result_1.get("answer", "")
    print(f"  答案: {answer_1[:200]}..." if len(answer_1) > 200 else f"  答案: {answer_1}")