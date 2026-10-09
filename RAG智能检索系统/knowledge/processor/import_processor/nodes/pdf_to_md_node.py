#内置包
import json,time
import subprocess
from pathlib import Path
from typing import Tuple
#三方包
#项目包
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import (
    ValidationError, FileProcessingError, PdfConversionError, StateFieldError
)

class PdfToMdNode(BaseNode):
    """
    PDF转Markdown节点

    调用MinerU命令行工具将PDF转换为Markdown
    """
    node_name = 'pdf_to_md_node'
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        执行PDF转换

        :param state: 图状态
        :return: 更新后的状态
        """
        import_file_path_obj,file_dir_obj = self._validate_state_inputs_path(state)

        processed_code = self._execute_mineru_parse(import_file_path_obj,file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError(message='mineru解析pdf文件失败',node_name=self.node_name)

        md_path = self.get_md_path(import_file_path_obj,file_dir_obj)

        state['md_path'] = md_path
        return state


    def _validate_state_inputs_path(self,state:ImportGraphState)->Tuple[Path,Path]:
        """
        验证输入路径

        :param state: 该节点接收到的状态
        :return: 文件路径及文件目录元组
        """
        self.log_step('Step1','对状态路径输入参数做校验')

        import_file_path = state.get('import_file_path','')
        file_dir = state.get('file_dir','')
        if not import_file_path:
            raise StateFieldError(node_name=self.node_name,field_name ='import_file_path',expected_type=str)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name='import_file_path', expected_type=str,
                                  message='解析文件路径不存在')

        if not file_dir:
            raise StateFieldError(node_name=self.node_name, field_name='file_dir', expected_type=str,)

        file_dir_obj = Path(file_dir)

        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name='file_dir', expected_type=str,
                                  message='输出目录不存在')

        self.logger.info(f"解析的文件路径{import_file_path}")
        self.logger.info(f"输出的文件目录{file_dir}")

        return import_file_path_obj,file_dir_obj

    def _execute_mineru_parse(self,import_file_path_obj:Path,file_dir_obj:Path)->int:
        """

        :param import_file_path_obj: 解析文件的path路径
        :param file_dir_obj: 解析后文件输出目录
        :return:状态（0或非0）
        0:成功的（底层封装状态码）
        非0：失败的
        """
        cmd = ['mineru','-p',str(import_file_path_obj),'-o',str(file_dir_obj),'--source','local']
        start_time = time.time()
        proc = subprocess.Popen(args=cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                errors='replace',
                                encoding='utf-8',
                                bufsize=1)
        for line in proc.stdout:
            self.logger.info(f"MinerU解析产生的日志：{line}")
        processed_result = proc.wait()
        end_time = time.time()
        if processed_result ==0:
            self.logger.info(f"mineru解析pdf文件成功，总耗时：{end_time-start_time:.2f}s")
        else:
            self.logger.info("mineru解析pdf文件失败")
        return processed_result

    def get_md_path(self,import_file_path_obj:Path,file_dir_obj:Path)->str:
        """
        获取解析后md的路径
        :param import_file_path_obj:
        :param file_dir_obj:
        :return:
        """
        file_name = import_file_path_obj.stem
        return  str(file_dir_obj/file_name/'hybrid_auto'/f'{file_name}.md')



##########################################
# 测试
##########################################

if __name__ == '__main__':
    setup_logging()
    # 1. 构建节点实例
    pdf_to_md_node = PdfToMdNode()

    # 2. 构建该节点状态
    init_state = {
        "import_file_path": r"G:\AI-project\shopkeeper_brain\knowledge\test\万用表RS-12的使用.pdf",
        "file_dir": r"G:\AI-project\shopkeeper_brain\knowledge\test"
    }

    # 3. 直接调用process
    result = pdf_to_md_node.process(init_state)

    # 4. 序列化（将对象转成字符串） 反序列化（将字符串转成对象）
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)


