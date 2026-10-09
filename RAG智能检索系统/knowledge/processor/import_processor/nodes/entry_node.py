#内置包
import json
from pathlib import Path

#三方包

#项目包
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import ValidationError

class EntryNode(BaseNode):
    """
    入口节点

    根据输入文件的拓展名设置相应的处理标志，
    决定后续流程走PDF转换分支还是直接处理MD分支，或者图片处理分支
    """

    node_name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        检测文件类型

        :param state: ImportGraphState 该节点处理之前的节点状态
        :return: ImportGraphState 该节点处理之后的节点状态
        """
        # 获取文件所在目录和导入文件路径
        self.log_step('Step1','[获取文件路径]')
        file_dir = state.get('file_dir')
        import_file_path = state.get('import_file_path')

        # 校验文件路径及目录
        self.log_step('Step2','[检测文件路径]')
        if not file_dir or not import_file_path:
            raise ValidationError(f'文件目录或文件不存在:',self.node_name)

        path = Path(import_file_path)   #Path 的主要作用不是“把地址变成绝对路径”，而是把字符串路径转换成可操作的路径对象，方便解析文件信息
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path

        elif suffix =='.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        file_title = path.stem
        state['file_title'] = file_title

        return state
if __name__ == '__main__':
    setup_logging()

    # 方式：直接实例该节点对象，调用 process 方法
    # 1. 构建该节点需要的 state
    test_entry_state = {
        "file_dir": r"D:\test_data",
        "import_file_path": r"D:\test_data\万用表的使用.pdf"
    }

    # 2. 实例 EntryNode 节点
    entry_node = EntryNode()

    # 3. 调用 process 方法
    processed_state = entry_node(test_entry_state)

    # 序列化打印
    print(json.dumps(processed_state, ensure_ascii=False, indent=4))

