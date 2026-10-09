import json
import os
import re
from typing import Dict,List,Any
# 三方包
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目包
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.base import BaseNode,setup_logging
from knowledge.processor.import_processor.config import get_config
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """
    主要职责：
    1、分割md文档内容
    2、按照标题、过长文段缩减、过短合并，三级策略分割
    """
    node_name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1、获取数据
        config = get_config()
        # 2、校验参数
        md_content,file_title,max_content_length,min_content_length=self._validate_state(state,config)
        # 3、按标题切割
        sections = self._split_by_headings(md_content,file_title)
        # 4、按内容长短切分，并合并
        split_chunks = self._split_and_merge(sections,max_content_length,min_content_length)
        # 5、组合最终chunks
        chunks = self._assemble_chunk(split_chunks)
        # 6、更新state
        state['chunks'] = chunks

        # 7、备份文件
        self._backup_chunks(state,chunks)

        return  state


    def _validate_state(self, state:ImportGraphState, config):
        self.log_step('【Step1】','切分文档参数获取及校验')

        md_content = state.get('md_content')

        if not md_content:
            raise ValueError(f"md文档内容不存在")

        file_title = state.get('file_title')

        if config.max_content_length <=0 or config.min_content_length<=0 or config.max_content_length <= config.min_content_length:
            raise ValueError(f'切片长度参数校验失败')
        return md_content,file_title,config.max_content_length,config.min_content_length



    def _split_by_headings(self, md_content:str, file_title:str)->List[Dict]:
        """
        主要职责：按标题切分文档
        Args:
            md_content: 文档内容
            file_title: 文件标题

        Returns:sections List

        """
        self.log_step('【Step2】','按标题切分文档')

        body_lines = []
        in_fence = False
        sections = []
        hierarchy = [""] * 7
        cur_title = ""
        cur_level = 0

        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        content_lines = md_content.splitlines()

        def _flush():
            """
            封装section对象
            Returns:
            """
            body = '\n'.join(body_lines)
            parent_title = ""
            if cur_title or body:
                for i in range(cur_level-1,0,-1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = cur_title if cur_title else file_title

                return sections.append({
                    'cur_title':cur_title if cur_title else file_title,
                    'parent_title':parent_title,
                    'file_title':file_title,
                    'body':body
                })


        for cur_content in content_lines:

            if cur_content.strip().startswith('```') or cur_content.strip().startswith('~~~'):
                in_fence = not in_fence

            match = heading_re.match(cur_content) if not in_fence else None

            if match:
                _flush()

                level = len(match.group(1).strip())
                cur_level = level
                cur_title = cur_content
                hierarchy[level] = cur_title



                for i in range(cur_level+1,7):
                    hierarchy[i] = ""
                body_lines = []
                print(hierarchy)

            else:
                body_lines.append(cur_content)

        _flush()
        return sections

    def _split_and_merge(self, sections:List[Dict], max_content_length:int, min_content_length:int)->List[Any]:
        """
        主要职责：分割长度长于最大文段长度得文段，最后合并切割后剩余过短的文段
        Args:
            sections: 按标题分割好的文段
            max_content_length: 最大长度限制
            min_content_length: 最小长度限制

        Returns:chunks

        """

        cur_section = []
        for section in sections:
            cur_section.extend(self.split_long_section(section,max_content_length))
        final_section = self.merge_short_section(cur_section,min_content_length)

        return final_section

    def split_long_section(self, section:Dict[str,Any], max_content_length:int)->List[Any]:
        """
        主要职责：分割body中长度长于最大长度限制的文段
        Args:
            section: 按标题切割好的文段
            max_content_length: 最大长度限制

        Returns:

        """
        cur_title = section.get('cur_title')
        parent_title =  section.get('parent_title')
        file_title = section.get('file_title')
        body = section.get('body')

        if "<table>" or '|' in body:
            body = MarkdownTableLinearizer.process(body)
            section['body'] =body
            

        if len(cur_title)>50:
            cur_title = cur_title[:50]

        total_length = len(f"{cur_title}\n\n") + len(body)

        if total_length <=max_content_length:
            return [section]

        body_length = max_content_length-len(f"{cur_title}\n\n")
        if body_length <=0:
            return [section]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n","\n","。","！","？","；",".",",",";","?","!"," ",""],
            keep_separator=True
        )

        split_text = text_splitter.split_text(body)
        if len(split_text) == 1:
            return [section]

        sub_section = []
        for index,text in enumerate(split_text):
            sub_section.append({
                "cur_title":f"{cur_title}-{index+1}",
                "file_title":file_title,
                "parent_title":parent_title,
                "body":text
            })

        return  sub_section

    def merge_short_section(self, cur_section:List[Any], min_content_length:int)->List[Any]:
        """
        主要职责：最大文段切割后，切割剩余文段长度小于最小长度限制的文段
        Args:
            cur_section: 过长文段切割后的整合文段
            min_content_length:最小长度限制

        Returns:过短文段合并后的最后整合文段

        """
        pre_section = cur_section[0]
        final_section = []

        for sub_section in cur_section[1:]:
            same_parent = (pre_section['parent_title'] == sub_section['parent_title'])
            if same_parent and len(pre_section.get('body')) < min_content_length:
                pre_section['body'] = pre_section.get('body').rstrip() +"\n\n" + sub_section.get('body').lstrip()

                pre_section['title'] = pre_section['parent_title']

            else:
                final_section.append(pre_section)
                pre_section = sub_section

        final_section.append(pre_section)

        return  final_section



    def _assemble_chunk(self,split_chunks):
        """
        主要职责：合并切割后，剩余文段长度小于最小长度限制的文段
        Args:
            split_chunks:

        Returns:

        """
        chunks = []
        for chunk in split_chunks:
            cur_title = chunk.get('cur_title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{cur_title}\n\n{body}"  # 标题注入到内容中

            chunks.append({
                "cur_title": cur_title,  # 用于检索结果展示
                "file_title": file_title,  # 用于溯源：来自哪个文档
                "parent_title": parent_title,  # 用于溯源：属于哪个章节
                "content": content,  # 用于向量嵌入和 LLM 上下文
            })
        return chunks

    def _backup_chunks(self, state:ImportGraphState, chunks:List[Any]):
        """
        备份文档
        Args:
            state: 更新后的状态
            chunks: 切割好的文档

        Returns:新文档

        """
        local_dir = state.get('file_dir','')
        if not local_dir:

            return
        try:
            os.makedirs(local_dir,exist_ok=True)
            output_path = os.path.join(local_dir,'chunks.json')
            with open(output_path,'w',encoding="utf-8") as f:
                json.dump(chunks,f,ensure_ascii=False,indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")



if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"G:\AI-project\shopkeeper_brain\knowledge\test\万用表RS-12的使用\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "测试文件",
        "md_content": content,
        "file_dir": r"G:\AI-project\shopkeeper_brain\knowledge\test\文档切分测试"
    }
    document_node.process(state)






