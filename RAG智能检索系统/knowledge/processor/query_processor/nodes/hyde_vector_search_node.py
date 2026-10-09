from typing import Tuple, List, Dict, Any,Union

from langchain_core.messages import HumanMessage,SystemMessage

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients,BGEM3EmbeddingFunction
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_utils import generate_bge_m3_hybrid_vector
from knowledge.utils.milvus_utils import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE


class HydeVectorSearchNode(BaseNode):
    """混合向量检索"""
    node_name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.log_step("【Step1】","参数校验")
        rewritten_query,item_names = self._validate_state(state)

        self.log_step("【Step2】", "获取milvus、BGE_M3、LLM客户端")
        try:
            llm_client = AIClients.get_llm_client()
            milvus_client = StorageClients.get_milvus_client()
            bge_m3_client:BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"客户端获取失败，原因：{str(e)}")

        self.log_step("【Step3】", "LLM生成假设答案")
        try:
            system_prompt = f"你是一维{item_names}方面的技术文档领域专家，主要擅长编写技术文档、操作手册、文档规格说明"
            user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names =item_names,rewritten_query = rewritten_query )
            llm_response=llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"LLM生成{item_names}的假设性文档失败 原因:{str(e)}")
        hyde_document_answer = llm_response.content.strip()

        if not hyde_document_answer :
            self.logger.error(f"LLM生成{item_names}的假设性文档不存在")
            return state


        self.log_step("【Step4】", "嵌入向量化")
        try:
            embed_hyde_query_vector=generate_bge_m3_hybrid_vector(model=bge_m3_client,embedding_documents=[f"{rewritten_query}\n{hyde_document_answer}"])
        except Exception as e:
            self.logger.error(f"假设性文档{hyde_document_answer}向量化嵌入失败，原因：{str(e)}")
            return state


        self.log_step("【Step5】", "创建混合搜索请求")

        try:
            # 创建过滤参数
            expr,expr_params = _item_names_filter(item_names)
            # 创建混合搜索请求
            hybrid_search_request = create_hybrid_search_requests(dense_vector=embed_hyde_query_vector["dense"][0],
                                          sparse_vector=embed_hyde_query_vector["sparse"][0],
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

            state["hyde_embedding_chunks"] = hybrid_search_res[0]
            return  state
        except Exception as e:
            self.logger.error(f"假设性文档{hyde_document_answer}执行混合搜索查询失败 原因:{str(e)}")
            return state

    def _validate_state(self, state:QueryGraphState):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query,str):
            raise StateFieldError(node_name=self.node_name,field_name="rewritten_query",expected_type=str)

        if not item_names or not isinstance(item_names,list):
            raise StateFieldError(node_name=self.node_name,field_name="item_names",expected_type=list)

        return rewritten_query,item_names









