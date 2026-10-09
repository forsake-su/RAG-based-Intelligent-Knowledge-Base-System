import asyncio
import json
from json import JSONDecodeError
from typing import Dict,List,Tuple,Any,Union
from agents.mcp import MCPServerStreamableHttp


from agents import Agent, Runner
from agents.model_settings import ModelSettings

from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError


class WebMcpSearchNode(BaseNode):
    """互联网MCP查询节点"""
    node_name  = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str, Any]]:

        self.log_step("【Step1】","参数校验")
        rewritten_query, item_names = self._validate_state(state)

        self.log_step("【Step2】", "定义并且执行mcp的调用")

        web_search_res=asyncio.run(self._execute_mcp_server(rewritten_query))

        if not web_search_res:
            return  state

        return {"web_search_docs": web_search_res}


    def _validate_state(self, state:QueryGraphState):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query,str):
            raise StateFieldError(node_name=self.node_name,field_name="rewritten_query",expected_type=str)

        if not item_names or not isinstance(item_names,list):
            raise StateFieldError(node_name=self.node_name,field_name="item_names",expected_type=list)

        return rewritten_query,item_names

    async def _execute_mcp_server(self, rewritten_query:str)->List[Dict[str,Any]]:
        """
        MCP服务
        Args:
            rewritten_query:

        Returns:

        """
        # 1. 定义MCP客户端(StreamableHttp方式)
        async with MCPServerStreamableHttp(
                name="联网搜索",  # MCP客户端名字
                params={  # 提供MCP服务的第三方平台的api_key和base_url
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                    "timeout": 60,  # 超时时间
                },
                cache_tools_list=True,  # 缓存MCP服务下的工具列表的,加速
                max_retry_attempts=3,  # 重试次数
        ) as mcp_client:
            # 2. 调用工具
            web_search_result = await mcp_client.call_tool(tool_name="bailian_web_search",
                                                           arguments={"query": rewritten_query, "count": 3})
            # 3. 解析数据
            # 3.1 获取文本内容块对象
            text_content = web_search_result.content[0]
            if not text_content:
                return []

            # 3.2 获取文本内容块对象的内容
            text_content_text = text_content.text
            if not text_content_text:
                return []

            # 3.3 反序列化
            try:
                text_content_obj: Dict[str, Any] = json.loads(text_content_text)

                # 3.4 获取真正的网页内容
                pages = text_content_obj.get('pages', [])
                if not pages:
                    return []

                # 3.5 遍历
                web_search_results = []
                for page in pages:
                    web_search_results.append({
                        "snippet": page.get('snippet', '').strip(),
                        "title": page.get('title', '').strip(),
                        "url": page.get('url', '').strip(),
                    })
                return web_search_results
            except JSONDecodeError as e:
                self.logger.error(f"web_search检索失败 失败信息：{e.msg} 失败的内容:{e.doc} 失败的位置：{e.pos}")
                return []


if __name__ == '__main__':
    web_search_node = WebMcpSearchNode()

    mock_state = {
        "rewritten_query": "今天天气如何",
        "item_names": ["RS-12 数字万用表"],
    }

    result = web_search_node.process(mock_state)
    print(result)



