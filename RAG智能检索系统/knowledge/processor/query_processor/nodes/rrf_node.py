from typing import List,Tuple,Dict,Any

from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode


class RrfNode(BaseNode):

    node_name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:

        self.log_step("【Step1】","校验数据")
        vector_search_chunks,hyde_search_chunks = self._validate_state(state)

        self.log_step("【Step2】","设置权重")
        search_source = {
            "vector_search_result":(self._normalize_input(vector_search_chunks),1.0),
            "hyde_search_result":(self._normalize_input(hyde_search_chunks),1.0),
        }

        self.log_step("【Step3】", "计算各路搜索结果得分")
        rrf_merge_results :List[Tuple[Dict[str,Any],float]] = self._rrf_merge(search_source,self._top_k,self._rrf_k)


        self.log_step("【Step4】", "格式化输出")

        rrf_chunks = [doc for doc,_ in rrf_merge_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新 state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回 state
        return state




    def _validate_state(self, state:QueryGraphState):
        vector_search_chunks = state.get("embedding_chunks",[])
        hyde_search_chunks = state.get("hyde_embedding_chunks",[])

        return vector_search_chunks,hyde_search_chunks

    def _normalize_input(self, search_chunks:List[Dict[str,Any]])->List[Dict[str,Any]]:
        """
        格式化处理各路检索结果
        Args:
            search_chunks: 各路检索结果

        Returns:格式化后的各路检索结果

        """
        path_result = []
        if not search_chunks or not isinstance(search_chunks,list):
            return []
        for chunk in search_chunks:
            if not isinstance(chunk,dict):
                continue

            entity = chunk.get("entity",{})
            if not entity or entity is None:
                continue
            path_result.append(entity)

        return  path_result

    def _rrf_merge(self, search_source:Dict[str,Any], _top_k, _rrf_k)->List[Tuple[Dict[str,Any],float]]:
        """

        Args:
            search_source: 各路搜索结果（格式化处理后）
            _top_k: 合并后最大返回数
            _rrf_k: 平滑参数K,默认为60

        Returns:合并以及排序后的文档列表

        """
        chunk_scores = {}
        chunk_data = {}

        rrf_input_value=list(search_source.values())
        for path_value,weight in rrf_input_value:
            for i,chunk in enumerate(path_value,1):
                chunk_id=chunk.get("chunk_id")
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id,0) + weight/(_rrf_k+i)

                # 去重 输出字典结构{chunk_id : chunk}
                chunk_data.setdefault(chunk_id,chunk)

        sorted_results = sorted([(chunk_data[cid],score)for cid,score in chunk_scores.items()],key = lambda x:x[1],reverse=True)
        print(sorted_results)
        return sorted_results[:_top_k] if _top_k else sorted_results




if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)
    print(result)

"""输出数据结构：
[
	({'chunk_id': 'chunk_1', 'content': '向量搜索结果#1'}, 0.03252247488101534), 
	({'chunk_id': 'chunk_2', 'content': '向量搜索结果#2'}, 0.03252247488101534), 
	({'chunk_id': 'chunk_3', 'content': '向量搜索结果#3'}, 0.015873015873015872), 
	({'chunk_id': 'chunk_4', 'content': 'HyDE搜索结果#3'}, 0.015873015873015872)
]
"""





