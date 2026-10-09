from typing import List,Dict,Union,Tuple,Any

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.task_util import set_task_result
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.sse_util import push_sse_event,SSEEvent
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.mongo_history_util import save_chat_message

class AnswerOutputNode(BaseNode):

    name = 'answer_output_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:

        task_id = state.get("task_id")
        is_stream = state.get('is_stream')
        # 1. 获取state已有答案
        if state.get('answer'):
            self._push_exist_answer(state,task_id)
            has_streamed = False
        # 2. 暂未有答案(经过三路检索完)
        else:
            prompt = self._build_prompt(state)
            state['prompt'] = prompt
            # 3 生成答案
            self._generate_answer(prompt,state)
            has_streamed = True

        self._save_history(state)

    def _push_exist_answer(self, state:QueryGraphState, task_id:str):
        """
        推送已存在答案至前端
        Args:
            state: 状态
            task_id:

        Returns:

        """
        if not state.get('is_stream'):
            set_task_result(state['task_id'],'answer',state.get('answer'))

    def _build_prompt(self, state:QueryGraphState)->str:
        """
        构建提示词
        Args:
            state:

        Returns:

        """
        char_budget = self.config.max_context_chars

        question = state.get('rewritten_query')
        item_names = state.get('item_names') or []

        format_context_char,char_budget = self._format_reranked_doc(state.get('reranked_docs')or [],char_budget)

        format_history_str,char_budget = self._format_history(state.get('history') or [],char_budget)

    def _format_reranked_doc(self, reranked_docs : List[Dict[str,Any]], char_budget : int)->Tuple[str,int]:
        """
        格式化reranker后的文档
        Args:
            reranked_docs: 重排序文档
            char_budget: 总字符预算

        Returns:格式化后的重排序文档,以及剩下可用字符

        """
        formatted_lines = []
        used_chars = 0
        for index,doc in enumerate(reranked_docs):

            content = doc.get('content','')
            if not content:
                continue
            content_meta = [f"[{index}]:"]
            for meta_field,template,fmt in [('source','[source={}]',lambda x:str(x).strip()),
                                        ('chunk_id','[chunk_id={}]',lambda x:str(x).strip()),
                                        ('url','[url={}]',lambda x:str(x).strip()),
                                        ('title','[title={}]',lambda x:str(x).strip()),
                                        ('score','[score={}]',lambda x:f"{float(x):.4f}")]:
                meta_field_value = doc.get(meta_field,'')
                if meta_field_value:
                    content_meta.append(template.format(fmt(meta_field_value)))

            doc_entry = " ".join(content_meta) + "\n" + content

            seperator_usage = 2 if formatted_lines else 0

            total_usage = seperator_usage +len(doc_entry)
            if used_chars + total_usage > char_budget:
                break
            used_chars +=total_usage
            formatted_lines.append(doc_entry)

        return "\n\n".join(formatted_lines) , char_budget - used_chars

    def _format_history(self, history :List[Dict[str,Any]], char_budget : int) -> Tuple[str,int]:
        """
        格式化历史对话记录
        Args:
            history:历史对话记录
            char_budget:剩余字符限制

        Returns:

        """
        formatted_lines = []
        used_chars = 0

        role_map = {'user' : "用户",'assistant':"助手"}
        for msg in history:
            role = msg.get('role','')
            text = msg.get('text','')

            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}:{text}"

            seperator_usage = 1 if formatted_lines else 0

            # 计算总长度
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return  '\n'.join(formatted_lines),char_budget-used_chars

    def _generate_answer(self, prompt:str, state:QueryGraphState):
        """
        调用llm生成答案
        Args:
            prompt:提示词
            state:

        Returns:LLM输出内容

        """

        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"LLM客户端调用失败，原因：{str(e)}")
            state["answer"] = "LLM客户端调用失败"
            return

        task_id = state.get('task_id')
        is_stream = state.get('is_stream')
        if is_stream:
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(llm_client, prompt)
            set_task_result(task_id, "answer", state["answer"])

    def _stream_generate(self, llm_client, prompt:str, task_id:str)->str:
        """
        流式生成
        Args:
            llm_client:llm客户端
            prompt:
            task_id:任务id

        Returns:

        """
        accumulate_answer = ''
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk,'content','') or ''
                if delta_text:
                    accumulate_answer += delta_text
                    push_sse_event(task_id = task_id,
                                   event=SSEEvent.DELTA,
                                   data={'dalta':delta_text})
        except Exception as e:
            self.logger.error(f"流式生成异常，原因{str(e)}")
        return accumulate_answer

    def _invoke_generate(self, llm_client, prompt:str)->str:
        try:
            res = llm_client.invoke(prompt)
            return res.content
        except Exception as e:
            self.logger.error(f"非流程生成异常，原因：{str()}")
            return "抱歉，生成回答时出现异常。"

    def _save_history(self, state:QueryGraphState):
        """
        保存对话记录
        Args:
            state:

        Returns:

        """
        user_query = state.get('rewritten_query')
        item_names = state.get('item_names') or []
        answer = state.get('answer')
        session_id = state.get('session_id')

        try:
            if answer:
                # 保存用户信息
                save_chat_message(session_id = session_id,
                                  role='user',
                                  text= state.get('original_query'),
                                  rewritten_query=user_query,
                                  item_names = item_names,
                                  message_id=None
                                  )
                # 保存助手信息
                save_chat_message(session_id=session_id,
                                  role="assistant",
                                  text=answer,
                                  rewritten_query=user_query,
                                  item_names=item_names,
                                  message_id=None)
        except Exception as e:
            self.logger.error(f"保存历史对话信息失败，原因：{str(e)}")
