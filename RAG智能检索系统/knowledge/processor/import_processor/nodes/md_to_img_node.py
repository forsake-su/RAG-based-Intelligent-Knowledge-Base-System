#内置包
import base64
import re
from collections import deque
from pathlib import  Path
from typing import Tuple, List, Optional,Dict,Set,Deque
from dataclasses import dataclass
from logging import Logger
import time

from ray._private.thirdparty.pathspec import pattern

#三方包

#项目包
from knowledge.processor.import_processor.base import BaseNode, T,setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """
    一张图片上下文信息
    """
    img_head:str
    pre_text:str
    post_text:str

@dataclass
class ImageInfo:
    """
    一张图片完整信息：
    图片id:
    图片地址：
    图片上下文信息：
    """
    img_name:str
    img_path:str
    img_content:ImageContext


class _MdFileHandler:

    def __init__(self,logger:Logger,node_name:str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state)->Tuple[str,Path,Path]:
        """
        读取及验证md文档方法：
        核心逻辑：
        1、读取md内容
        2、读取md路径
        3、获取图片目录
        Args:
            state: 上一节点更新的state

        Returns:md_content,md_path_obj,img_dir_obj
        """
        md_path = state.get('md_path')
        # 验证路径是否为空
        if not md_path:
            raise StateFieldError(node_name=self.node_name,field_name=md_path,expected_type=str)

        md_path_obj = Path(md_path)
        # 验证路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name,field_name=str(md_path_obj),expected_type=str,message='md文件路径不存在')

        try:
            with open(md_path_obj,'r',encoding="utf-8") as f:
                md_content = f.read()
            self.logger.info(f"MD文件：{md_path_obj.stem}读取成功")

        except IOError as e:
            self.logger.error(f"MD文件：{md_path_obj.stem}读取失败")
            raise FileProcessingError(message='文件读取失败',node_name=self.node_name)

        img_dir = md_path_obj.parent/'images'
        img_dir_obj = Path(img_dir)
        return  md_content,md_path_obj,img_dir_obj

    def backup(self, md_path_obj:Path, new_md_content:str)->str:
        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImgScanner:
    """
    照片扫描：主要职责：
    1、根据图片目录，得到该目录下有效的图片文件
    2、在md文件中定位图片位置
    3、获取该图片的上下文内容
    4、组装所有图片的上下文内容
    """
    def __init__(self,logger:Logger):
        self.logger = logger

    def scan_img_dir(self,
                     img_dir_obj:Path,
                     md_content:str,
                     img_suffix:str,
                     img_content_length:int)->List[ImageInfo]:
        """
        核心逻辑：
        1、扫描指定图片目录下的所有图片文件
        2、遍历图片并去MD文件中定位位置
            2.1 上文信息（标题+上文内容）
            2.2 下文信息（下文内容）
        3、将每张图片的上下文信息封装为每张图片的完整信息到ImageInfo容器中
        Args:
            img_dir_obj:图片目录
            md_content:md文档内容
            img_suffix:图片后缀
            img_content_length:图片上下文信息限制长度

        Returns:List[ImageInfo]

        """
        img_info_list = []
        for img_path in img_dir_obj.iterdir():

            if not img_path.is_file():
                self.logger.error(f"{img_path}为无效文件")
                continue

            if  img_path.suffix not in img_suffix:
                self.logger.error(f'{img_path.suffix}为无效后缀')
                continue

            img_context = self._find_img_context(img_path.name,md_content,img_content_length)
            if not img_context:
                self.logger.error(f'在md文件中未找到该图片{img_path.name}的上下文')
                continue

            img_info_list.append(ImageInfo(
                img_name=img_path.name,
                img_path = str(img_path),
                img_content=img_context
            ))
        return img_info_list

    def _find_img_context(self, img_path_name:str, md_content:str, img_content_length:int)->Optional[ImageContext]:
        """
        查找图片上下文
        Args:
            img_path_name: 图片名称
            md_content: md文档
            img_content_length: 最大长度限制

        Returns:Optional[ImageContext]
        """
        pattern = re.compile(r'!\[.*?\]\(.*?'+re.escape(img_path_name)+r'.*?\)')
        md_lines = md_content.split('\n')

        for md_idx,md_line in enumerate(md_lines):
            if not pattern.search(md_line):
                continue

            img_head,head_index = self._find_heading_up(md_lines,md_idx)
            pre_lines = md_lines[head_index+1:md_idx]   #列表切分，内容按照索引号切分
            pre_context = self._extract_limited_context(pre_lines,img_content_length,direction='front')

            #搜索下文及标题
            end_index = self._find_heading_down(md_lines,md_idx)
            next_lines = md_lines[md_idx+1:end_index]
            post_context = self._extract_limited_context(next_lines,img_content_length,direction='back')
            return ImageContext(
                img_head = img_head,
                pre_text=pre_context,
                post_text=post_context
            )
        return None

    def _find_heading_up(self, md_lines:List[str], md_idx:int)->Tuple[str,int]:
        """
        寻找图片上标题及索引号
        Args:
            md_lines: md文档内容
            md_idx: 图片索引号

        Returns:当前图片最近的上标题及索引号

        """

        for i in range(md_idx-1,-1,-1):
            if re.match(r'^#{1,6}\s+',md_lines[i]):  #match只取第一个遇到的与正则符合的片段，即标题
                return md_lines[i],i
        return '',-1

    def _find_heading_down(self, md_lines:List[str], md_idx:int)->int:
        """
        寻找图片下标题及索引号
        Args:
            md_lines: md文档内容
            md_idx: 图片索引号

        Returns:当前图片最近的下标题及索引号

        """
        for i in range(md_idx+1,len(md_lines)):
            if re.match(r'^#{1,6}\s+',md_lines[i]):
                return i
        return len(md_lines)

    def _extract_limited_context(self, extracted_md_lines, img_content_length, direction)->str:
        """
        主要职责：
        截取给定的上下文内容，采用段落方式截取，最后根据段落字数是否达到最大上下文长度选择留取段落的规则
        Args:
            extracted_md_lines:图片上下文内容
            md_idx:当前图片索引号
            img_content_length:最大长度限制
            direction:方向

        Returns:
            str:上下文内容
        """
        paragraphs = []
        cur_paragraph = []
        for line in extracted_md_lines:
            #是否为空行
            is_blank_line = not line.strip()
            #是否为图片段
            is_img_line = re.match(r"^!\[.*?\]\(.*?\)$",line.strip())

            if is_img_line or is_blank_line:
                if cur_paragraph:
                    paragraphs.append('\n'.join(cur_paragraph))
                    cur_paragraph = []
                continue
            cur_paragraph.append(line)

        if cur_paragraph:
            paragraphs.append('\n'.join(cur_paragraph))

        if direction =='front':
            paragraphs.reverse()
        total = 0
        collected_para = []
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and collected_para:
                break
            collected_para.append(paragraph)
            total +=len(paragraph)  # 当首个段落字数超出限制时，由于select_para为空, 即为False根据and的短路求值，左侧为True，右侧False，总体则为False，所以不会触发条件判断。

        if direction =='front':
            collected_para.reverse()
        # 4. 将最终段落列表中的段落转成一个字符串
        return "\n\n".join(collected_para)


class _VLMSummarizer:
    """
    主要职责：
    根据每一张图片信息以及上下文信息，生成该图片的对应摘要信息
    """
    def __init__(self,
                 logger:Logger,
                 request_per_minute:int):
        self.logger = logger
        self.request_per_minute = request_per_minute

    def _sum_abstract(self,document_name,img_info_list:List[ImageInfo],vlm_model:str)-> Dict[str, str]:
        """
        为所有图片生成摘要
        Args:
            document_name:文件名
            img_info_list: 图片信息
            vlm_model: vlm模型

        Returns:Dict[str,str]:{"img_name":"summary"}

        """
        summaries = {}
        request_timestamps:Deque[float] = deque()

        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            self.logger.error(f"VLM客户端初始化失败：{e}")
            for img_info in img_info_list:
                summaries[img_info.img_name] = '暂无摘要'
            return summaries

        for img_info in img_info_list:
            self._enforce_rate_limit(request_timestamps,self.request_per_minute)
            summaries[img_info.img_name] = self._sum_per_abstract(document_name,img_info,vlm_client,vlm_model)
        self.logger.info(f"{len(summaries)}个图片摘要")
        return summaries

    def _sum_per_abstract(self, document_name, img_info, vlm_client, vlm_model)->str:
        """
        根据每张图片上下文总结图片摘要
        Args:
            document_name: 文档名称
            img_info: 图片信息
            vlm_client: VLM客户端
            vlm_model: VLM模型
        Returns:图片摘要
        """
        context = [p for p in (img_info.img_content.img_head,img_info.img_content.pre_text,img_info.img_content.post_text) if p]
        final_context = '\n'.join(context) if context else '暂无上下文'
        try:
            with open(img_info.img_path,'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f'读取图片文件地址{img_info.img_path}失败：{e}')
            return '读取图片文件地址失败'

        try:
            resp = vlm_client.chat.completions.create(
                model=vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"



    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)




class _ImgUpLoader:
    """
        主要职责：
        1. 将本地图片上传到MinIO，得到该图片在MinIO中可访问的远程地址
        2. 替换md中的摘要和图片地址
        """

    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self,
                           document_name:str,
                           md_content:str,
                           img_summaries:Dict[str, str],
                           img_list:List[ImageInfo],
                           minio_bucket:str,
                           minio_base_url:str)->str:

        remote_urls = self._upload_all(document_name,img_list,minio_bucket,minio_base_url)

        new_md_content = self._replace_md(md_content,img_summaries,remote_urls)
        return new_md_content

    def _upload_all(self, document_name:str, img_list:List[ImageInfo], minio_bucket:str, minio_base_url:str)->Dict[str,str]:

        remote_urls = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.warning(f'Minio客户端加载失败')

        self.logger.info(f'Minio客户端加载成功')

        for img in img_list:
            object_name = f"{document_name}/{img.img_name}"
            try:
                minio_client.fput_object(
                    minio_bucket, object_name, img.img_path,
                )
                remote_url = (f"{minio_base_url}/{minio_bucket}/{object_name}")
                self.logger.info(f"{img.img_name}上传成功")
                remote_urls[img.img_name] = remote_url
            except Exception as e:
                self.logger.warning(f"{img.img_name}上传失败，保留本地路径")
                # remote_urls[img.img_name] = img.img_path
        self.logger.info(f"成功上传{len(remote_urls)}张图片至Minio")
        return remote_urls




    def _replace_md(self, md_content:str, img_summaries:Dict[str,str], remote_urls:Dict[str,str])->str:
        """
        替换原md文档中图片信息摘要为远程 URL + 摘要
        Args:
            md_content: 原md文档
            img_summaries: 图片摘要
            remote_urls:远程 URL
        Returns:新md文档
        """
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match:re.Match)->str:
            original_path = match.group(2).strip()
            img_name_in_md = Path(original_path).name
            try:
                for img_name,summary in img_summaries.items():
                    if img_name == img_name_in_md:
                        return f"![{summary}]({remote_urls[img_name] if remote_urls[img_name] else original_path })"
            except Exception as e:
                self.logger.warning(f"图片摘要替换失败")
            self.logger.info(f"成功替换图片摘要，新文件")
            return match.group(0)    #保证“这张图片没被处理”时，原封不动地保留它，避免文档内容丢失或损坏

        return pattern.sub(replacer, md_content)




# -------------主节点-------------

class MarkdownToImgNode(BaseNode):
    """
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法
    """

    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(self.logger,self.node_name)
        self._img_scaner = _ImgScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._img_uploader = _ImgUpLoader(self.logger)

    node_name = 'md_to_img_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        md文档及图片处理入口
        Args:
            state:进入该节点前的状态

        Returns:更新后的状态

        """
        config = self.config

        # 1. 读取MD内容、路径以及图片目录
        self.log_step('【Step_1】','读取MD内容、路径以及图片目录')
        md_content,md_path_obj,img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        if not img_dir_obj.exists():
            state['md_content'] = md_content
            self.logger.warning(f"图片文件夹{img_dir_obj}：不存在")
            return  state

        # 2. 扫描图片目录,获得图片上下文
        self.log_step('【Step_2】','扫描图片目录')
        img_info_list:List[ImageInfo] = self._img_scaner.scan_img_dir(img_dir_obj,md_content,config.image_extensions,config.img_content_length)

        # 3.VLM提炼图片摘要
        self.log_step('【Step_3】', 'VLM提炼图片摘要')
        summaries: Dict[str, str] = self._vlm_summarizer._sum_abstract(md_path_obj.stem, img_info_list, config.vl_model)

        # 4. 操作_img_uploader，上传图片，更新替换md文档图片摘要
        self.log_step('【Step_4】', '上传图片至Minio,更新md文档图片摘要')
        new_md_content = self._img_uploader.upload_and_replace(document_name=md_path_obj.stem,
                                                               md_content = md_content,
                                                               img_summaries = summaries,
                                                               img_list = img_info_list,
                                                               minio_bucket = config.minio_bucket,
                                                               minio_base_url = config.get_minio_base_url()
                                                               )

        # 5. 备份文件
        self.log_step("【Step_5】",'备份新文件')
        self._md_file_handler.backup(md_path_obj, new_md_content)

        # 6. 更新并返回state
        self.log_step("【Step_6】",'更新状态，返回state')
        state["md_content"] = new_md_content
        return state





        return state

if __name__ == '__main__':
    setup_logging()
    md_img_scan_node = MarkdownToImgNode()
    init_state = {
        "md_path": r"G:\AI-project\shopkeeper_brain\knowledge\test\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"
    }
    md_img_scan_node.process(init_state)