from typing import Tuple, List, Dict, Any,Union



from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients,BGEM3EmbeddingFunction
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_utils import generate_bge_m3_hybrid_vector
from knowledge.utils.milvus_utils import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter


class HybridVectorSearchNode(BaseNode):
    """混合向量检索"""
    node_name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.log_step("【Step1】","参数校验")
        rewritten_query,item_names = self._validate_state(state)

        self.log_step("【Step2】", "获取milvus和BGE_M3客户端")
        try:
            milvus_client = StorageClients.get_milvus_client()
            bge_m3_client:BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"客户端获取失败，原因：{str(e)}")

        self.log_step("【Step3】", "嵌入向量化")
        try:
            embed_query_vector=generate_bge_m3_hybrid_vector(model=bge_m3_client,embedding_documents=[rewritten_query])
        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}向量化嵌入失败，原因：{str(e)}")


        self.log_step("【Step4】", "创建混合搜索请求")

        try:
            # 创建过滤参数
            expr,expr_params = _item_names_filter(item_names)
            # 创建混合搜索请求
            hybrid_search_request = create_hybrid_search_requests(dense_vector=embed_query_vector["dense"][0],
                                          sparse_vector=embed_query_vector["sparse"][0],
                                          expr = expr,
                                          expr_params = expr_params,
                                          limit= 5
                                          )

            self.log_step("【Step5】", "执行混合搜索请求")

            hybrid_search_res =execute_hybrid_search_query(milvus_client= milvus_client,
                                        collection_name= self.config.chunks_collection,
                                        search_requests= hybrid_search_request,
                                        ranker_weight= [0.5,0.5],
                                        norm_score=True,
                                        limit=5,
                                        output_fields= ["chunk_id", "content", "item_name", 'title']
                                        )

            if not hybrid_search_res or not hybrid_search_res[0]:
                return state

            state["embedding_chunks"] = hybrid_search_res[0]
            return  state
        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}执行混合搜索查询失败 原因:{str(e)}")
            return state

    def _validate_state(self, state:QueryGraphState):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query,str):
            raise StateFieldError(node_name=self.node_name,field_name="rewritten_query",expected_type=str)

        if not item_names or not isinstance(item_names,list):
            raise StateFieldError(node_name=self.node_name,field_name="item_names",expected_type=list)

        return rewritten_query,item_names









