import os,json,logging
from typing import List,Dict,Tuple,Any,Optional
from pymilvus import MilvusClient,WeightedRanker,AnnSearchRequest, DataType, Function, FunctionType

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# 创建混合搜索请求
# ----------------------------------------------------------------

def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params = None,
        sparse_params = None,
        expr = None,
        expr_params = None,
        limit = 5)->List[AnnSearchRequest]:

    """
    创建混合搜索请求
    Args:
        dense_vector: 稠密向量
        sparse_vector: 稀疏向量
        dense_params: 稠密向量搜索参数
        sparse_params: 稀疏向量搜索参数
        expr: 过滤条件即查询表达式
        expr_params: 查询表达式的变量内容
        limit: 返回数量限制，默认为5

    Returns:包含稠密和稀疏搜索请求的列表

    """
    # 参数校验
    if dense_vector is None or sparse_vector is None:
        raise ValueError(f"dense_vector和sparse_vector不能为空")

    try:
        if dense_params is None:
            dense_params = {"metric_type":"COSINE"}
        if sparse_params is None:
            sparse_params = {"metric_type":"IP"}

        dense_request = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param = dense_params,
            expr = expr,
            expr_params= expr_params,
            limit = limit
        )

        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit
        )
        return [dense_request,sparse_request]
    except Exception as e:
        raise RuntimeError(f"创建混合搜索请求失败，原因：{str(e)}")

# ----------------------------------------------------------------
# 执行混合搜索请求
# ----------------------------------------------------------------


def execute_hybrid_search_query(
        milvus_client:MilvusClient,
        collection_name,
        search_requests,
        ranker_weight:list = [0.5,0.5],
        norm_score:bool = True,
        limit:int = 5,
        output_fields = None,
        search_params = None,
):
    """
    执行混合搜索
    Args:
        milvus_client: Milvus客户断
        collection_name: 集合名称
        search_requests: 搜索请求列表：[dense_req, sparse_req]
        ranker_weight: 权重排名器的权重，默认为(0.5, 0.5)
        norm_score: 是否对分数进行归一化，默认为True
        limit: 返回结果数量限制，默认为5
        output_fields: 要返回的字段列表，默认为None
        search_params: 搜索参数，默认为None

    Returns:

    """

    # 参数校验
    if milvus_client is None:
        raise ValueError("Milvus_client不能为空")

    if search_requests is None or len(search_requests) ==0:
        raise ValueError("search_requests不能为None或为空列表")

    try:
        weighted_rerank = Function(
            name='weighted',
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                "reranker":"weighted",
                "weights":ranker_weight,
                "norm_score":norm_score
            }
        )
        # 默认输出字段
        if output_fields is None:
            output_fields = ["item_name"]

        res = milvus_client.hybrid_search(
            collection_name = collection_name,
            reqs=search_requests,
            ranker= weighted_rerank,
            limit= limit,
            output_fields=output_fields,
            search_params = search_params
        )
        total_hits = sum(len(hits) for hits in res) if res else 0
        logger.info(f"milvus混合检索完成，共处理{len(res) if res else 0}个查询，共计找到{total_hits}个结果")
        return res
    except Exception as e:
        raise RuntimeError(f"执行Milvus混合搜索失败 (collection={collection_name}): {e}") from e

def _item_names_filter(item_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    expr = "item_name in {item_names}"
    expr_params = {"item_names": item_names}
    return expr, expr_params





















