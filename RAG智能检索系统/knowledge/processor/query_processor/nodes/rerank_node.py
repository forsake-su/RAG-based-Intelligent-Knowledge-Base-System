from typing import List,Dict,Tuple,Any
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.config import get_config
from knowledge.processor.query_processor.base import BaseNode
from knowledge.utils.bge_rerank_util import get_rerank_model


class RerankNode(BaseNode):

    node_name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        self.log_step("【Step1】","校验数据")

        rewritten_query=self._validate_state(state)

        self.log_step("【Step2】", "合并多源文档")
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_source_docs(state)

        self.log_step("【Step3】", "Rerank精排")
        reranked_docs=self._rerank_merged_docs(rewritten_query,merged_multi_docs)

        state["reranked_docs"] = reranked_docs


        return state

    def _validate_state(self, state:QueryGraphState):
        rewritten_query=state.get("rewritten_query","")
        if not rewritten_query :
            return state.get('original_query', '')

        return rewritten_query

    def _merge_multi_source_docs(self, state:QueryGraphState)->List[Dict[str, Any]]:
        """
        合并各路检索结果文档
        Args:
            state:

        Returns:

        """
        final_docs = []
        rrf_chunks = state.get("rrf_chunks",[])
        for rrf_doc in rrf_chunks:
            if not rrf_doc or not isinstance(rrf_doc,dict):
                continue
            content = rrf_doc.get("content","").strip()
            if not content:
                continue
            title = rrf_doc.get("title","").strip()
            chunk_id = rrf_doc.get("chunk_id",0)
            format_rrf_doc = self._format_rrf_docs(content= content,title = title,chunk_id = chunk_id,source = "local")
            final_docs.append(format_rrf_doc)

        web_chunks = state.get("web_search_docs",[])
        for web_doc in  web_chunks:
            if not isinstance(web_doc,dict):
                continue
            content = web_doc.get("snippet","").strip()
            if not content:
                continue

            title = web_doc.get("title","").strip()
            url = web_doc.get("url","").strip()
            format_web_doc = self._format_rrf_docs(content = content,title=title,chunk_id = None,url=url,source="web")
            final_docs.append(format_web_doc)

        self.logger.info(f"收集到准备进行 Rerank 精排的文档 {len(final_docs)}")
        return final_docs



    def _format_rrf_docs(self, content:str, title:str, chunk_id:Any, source:str ="",url:str = ""):
        return {
            "content":content,
            "title":title,
            "chunk_id":chunk_id,
            "source":source,
            "url":url
        }

    def _rerank_merged_docs(self, rewritten_query:str, merged_multi_docs:List[Dict[str,Any]])->List[Dict[str,Any]]:
        """
        精排
        Args:
            rewritten_query: 改写后的用户问题
            merged_multi_docs: 多路合并的搜索结果

        Returns:

        """
        if not merged_multi_docs:
            return []


        rerank_model = get_rerank_model()
        if rerank_model is None:
            self.logger.error("重排序模型Rerank获取失败")
            return []

        query_and_content_pairs = [(rewritten_query,doc.get("content","")) for doc in merged_multi_docs]

        try:
            rerank_score=rerank_model.compute_score(sentence_pairs=query_and_content_pairs)

            if isinstance(rerank_score, (float, int)):
                rerank_score = [rerank_score]

            score_doc = [{**doc,"score":score} for doc,score in zip(merged_multi_docs,rerank_score)]
            sorted_score_doc = sorted(score_doc,key=lambda x:x.get("score"),reverse=True)
            print(sorted_score_doc)
            return sorted_score_doc

        except Exception as e:
            self.logger.error(f"多路合并文档进行重排序失败，原因：{str(e)}")
            return [{**doc, "score": None} for doc in merged_multi_docs]

# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")


"""输出数据格式
[
    {
        "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。",
        "title": "主板维修手册",
        "chunk_id": "local_1",
        "source": "local",
        "url": "",
        "score": 5.0625
    },
    {
        "content": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。",
        "title": "短路查修指南",
        "chunk_id": null,
        "source": "web",
        "url": "https://example.com/repair",
        "score": 3.96484375
    },
    {
        "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。",
        "title": "闲聊",
        "chunk_id": "local_2",
        "source": "local",
        "url": "",
        "score": -4.7421875
    },
    {
        "content": "苹果发布新款手机，A系列芯片性能提升20%。",
        "title": "科技新闻",
        "chunk_id": null,
        "source": "web",
        "url": "https://example.com/news",
        "score": -9.4765625
    }
]
"""

