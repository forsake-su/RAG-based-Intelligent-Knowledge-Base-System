import logging
import re,json,logging
from json import JSONDecodeError
from typing import Dict,List,Tuple,Any
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

from knowledge.processor.query_processor.config import get_config
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.mongo_history_util import get_recent_messages
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_USER_TEMPLATE,ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.milvus_utils import create_hybrid_search_requests,execute_hybrid_search_query
from knowledge.utils.embedding_utils import generate_bge_m3_hybrid_vector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItemNameConfirmedNode(BaseNode):

    node_name = 'item_name_confirmed_node'

    def __init__(self):
        super().__init__()
        self._aligner = _ItemNameAligner()
        self._extractor = _ItemNameExtractor()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        主要职责：
        1、利用LLM改写用户问题及提取商品名关键词
        1.1、若提取到关键词，第二步
        1.2、若没提取到关键词，则返回UNKNOW

        2、根据关键词，在Milvus中提取商品名
        3、分路：
        3.1、若提取出精确关键词--->检索
            若提取模糊关键词--->提交给用户确认
            若未提取出关键词--->返回抱歉
        4、根据检索结果，结合LLM生成答案
        Args:
            state:初始状态

        Returns:更新后状态

        """

        self.log_step("【Step1】","获取用户问题")
        original_query = state.get("original_query")
        session_id = state.get('session_id')

        self.log_step("【Step2】", "获取历史对话")

        history_context = get_recent_messages(session_id =session_id,limit = 10 )
        formatted_history = []
        for history in history_context:
            role = history.get('role','')
            text = history.get('text','')
            formatted_context = f"角色：{role},内容：{text}"
            formatted_history.append(formatted_context)   # [str,str,....]
        formatted_history_str = ' '.join(formatted_history)

        self.log_step("【Step3】", "调用LLM模型，改写用户问题并提取商品名")
        llm_result:Dict[str,Any]=self._extractor.extract_item_name(original_query,formatted_history_str)

        item_names = llm_result.get("item_names")
        rewritten_query = llm_result.get("rewritten_query")

        self.log_step("【Step4】", "检索数据库并对齐")

        if item_names:
            confirmed,options = self._aligner.search_and_align(item_names)
        else:
            confirmed,options = [],[]

        self.log_step("【Step5】", "决策路由")
        self._decide(confirmed, options, state, rewritten_query, item_names)

        state['history'] = history_context
        return state


    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState,
                rewritten_query: str, item_names: List[str]):
        """
        根据confirmed、options来判断是继续检索还是返回用户提示信息
        Args:            confirmed:  已经确认的商品名列表
            options: 模糊的商品名列表
            state: 查询状态
            rewritten_query: 重写后的问题
            item_names: LLM提取到的商品名列表

        Returns:
        """
        if confirmed:
            state['item_names'] = confirmed  # 对齐后的商品名
            state['rewritten_query'] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


class _ItemNameExtractor:
    """商品名提取与问题改写"""

    def extract_item_name(self,original_query:str,history_context:str)->Dict[str,Any]:
        """
        提取商品名
        Args:
            original_query: 用户原始问题
            history_context: 历史上下文

        Returns:商品名列表 ---> {"item_name":[ste,str],"rewritten_query":str}

        """
        llm_result = {"item_names":[],
                      "rewritten_query":original_query
                      }

        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f"LLM客户端调用失败，原因：{str(e)}")
            return llm_result

        item_name_user_prompt = ITEM_NAME_EXTRACT_USER_TEMPLATE.format(
            history_text = history_context.strip() if history_context else "暂无用户上下文",
            query = original_query
        )

        try:
            llm_res = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=item_name_user_prompt)
            ])
        except Exception as e:
            logger.error(f"LLM客户端提取信息失败，原因：{str(e)}")
            return llm_result

        llm_res_content = llm_res.content

        if not llm_res_content:
            return llm_result

        parsed_result:Dict[str,Any] = self._clean_and_parse(llm_res_content)

        llm_result["item_names"] = parsed_result.get("item_names")
        llm_result["rewritten_query"] = parsed_result.get("rewritten_query") if parsed_result.get("rewritten_query") else original_query

        return llm_result




    def _clean_and_parse(self, llm_res_content:str)->Dict[str,Any]:
        """
        清洗并解析LLM输出的结果
        清洗：先去除首位空白，并显示去除可能存在的BOM头
        Args:
            llm_res_content: llm的输出

        Returns:

        """
        cleaned = llm_res_content.strip().lstrip('\ufeff')
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$",'',cleaned)

        json_pattern = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])",cleaned)

        if json_pattern:
            content = json_pattern.group(1)
        else:
            raise ValueError("未找到有效的JSON结构")

        try:
            llm_content_obj = json.loads(content)

            raw_item_names = llm_content_obj.get("item_names")
            if not isinstance(raw_item_names,list):
                item_names = []
            else:
                item_names = [item_name.strip() for item_name in raw_item_names if isinstance(item_name,str) and item_name]

            raw_rewritten_query = llm_content_obj.get("rewritten_query")

            if not isinstance(raw_rewritten_query,str):
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.strip()

            return {
                "item_names":item_names,
                "rewritten_query":rewritten_query
            }
        except JSONDecodeError as e :
            logger.error(f"LLM输出结果{llm_res_content}反序列化失败，原因：{str(e)}")
            raise JSONDecodeError(msg=e.msg,
                                  doc=e.doc,
                                  pos=e.pos)


class _ItemNameAligner:
    """与milvus数据库对齐数据"""

    def __init__(self):
        self._config = get_config()

    def search_and_align(self,item_names:List[str])->Tuple[List[str],List[str]]:
        """
        检索向量数据库并和向量数据库中的商品名对齐，
        Args:
            item_names: LLM提取出的商品名列表

        Returns:确定的商品名列表或者模糊的商品名列表

        """

        search_result = self._search_vector(item_names)
        if not search_result:
            return [],[]

        confirmed,options = self._align(search_result)

        if len(confirmed)>1:
            confirmed = self._item_name_score_filter(confirmed,search_result)

        return confirmed,options

    def _search_vector(self, item_names:List[str])->List[Dict[str,Any]]:
        """
        对商品名进行检索
        Args:
            item_names: 商品名列表

        Returns:检索结果：[{
                "extracted_name": item_name,
                "matches": matches  }]

        """
        final_search_result = []
        try:
            milvus_client = StorageClients.get_milvus_client()
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"客户端获取失败 原因:{str(e)}")
            return final_search_result

        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vector(model = bge_m3_client,
                                                                 embedding_documents=item_names)
        except Exception as e:
            logger.error(f"商品列表{item_names}向量化嵌入失败，原因：{str(e)}")
            return final_search_result

        for index,item_name in enumerate(item_names):
            hybrid_requests = create_hybrid_search_requests(dense_vector = hybrid_vector_result["dense_vector"][index],
                                                            sparse_vector= hybrid_vector_result["sparse_vector"][index]
                                                            )
            hybrid_search_result = execute_hybrid_search_query(milvus_client=milvus_client,
                                                               collection_name=self._config.item_name_collection,
                                                               search_requests=hybrid_requests,
                                                               ranker_weight=[0.5,0.5],
                                                               norm_score=True,
                                                               limit=5,
                                                               output_fields=['item_name']
                                                               )

            matches = [{"score":item_name_search_res['distance'],
                        "item_name":item_name_search_res['entity']['item_name']
                        } for item_name_search_res in (hybrid_search_result[0] if hybrid_search_result else []) ]

            final_search_result.append({"extracted_name":item_name,
                                        "matches":matches
                                        })
        return final_search_result

    def _align(self, search_result:List[Dict[str,Any]]):
        """
        主要职责：根据检索出的结果进行对齐
        Args:
            search_result:向量数据库检索到的结果

        Returns:最终两个容器confirmed、options列表中的商品名

        """
        confirmed = []
        options = []

        for item_search_result in search_result:
            # 获取LLM提取出的商品名
            llm_extract_item_name = item_search_result.get("extracted_name")
            # 获取从MIlvus中检索出的商品名
            milvus_item_name_matches = item_search_result.get('matches')
            milvus_item_name_matches = sorted(milvus_item_name_matches,key=lambda x :x['score'],reverse=True)

            high_score_item = [h for h in milvus_item_name_matches if h['score'] > self._config.item_name_high_confidence]

            if high_score_item:
                extract = next((h for h in high_score_item if h['item_name'] == llm_extract_item_name ),None)

                if extract:
                    picked = extract.get('item_name')
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high_score_item) ==1:
                    picked = high_score_item[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    top_score = high_score_item[0]['score']
                    if top_score-high_score_item[1]['score'] >= self._config.item_name_score_gap:
                        picked = high_score_item[0]['item_name']
                        if picked not in confirmed:
                            confirmed.append(picked)
                    else:
                        for h in high_score_item[:self._config.item_name_max_options]:
                            picked = h.get('item_name')
                            if picked not in options and picked not in confirmed:
                                options.append(picked)
            else:
                mid = [m for m in milvus_item_name_matches if m['score'] >= self._config.item_name_mid_confidence
                       and m.get('item_name') not in options and m.get('item_name') not in confirmed]

                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get("item_name"))

        return confirmed, options[:self._config.item_name_max_options]


    def _item_name_score_filter(self, confirmed, search_result):
        # 1. 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for search_result in search_result:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

                    # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed        if not item_name_score:
            return confirmed

            # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if
                max_score - score <= self._config.item_name_score_gap]



if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        # "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        "original_query": "PS PRO RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)









