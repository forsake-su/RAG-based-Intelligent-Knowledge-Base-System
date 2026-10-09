# 内置包
import numbers
from typing import Any,Dict,List,Tuple,Optional
import  json
import math
from collections.abc import Mapping,Sequence

# 三方包
from pymilvus import MilvusClient,DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_core.messages import HumanMessage,SystemMessage


# 项目包
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError,StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT,ITEM_NAME_USER_PROMPT_TEMPLATE


class ItemNameRecognitionNode(BaseNode):
    """商品名识别节点"""
    node_name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1、参数校验
        file_title,chunks,item_name_chunks_k,item_name_chunk_size=self._validate_state(state)


        # 2、提取商品名识别上下文
        item_name_context = self._item_name_recognition_context(chunks,item_name_chunks_k,item_name_chunk_size)

        # 3、调用LLM模型，提取商品名
        item_name = self._recognition_item_name(item_name_context,file_title)

        # 4、将提取的商品名利用BGE-M3进行向量化嵌入

        dense_vector,sparse_vector = self._embedding_item_name(item_name)

        # 5、入库milvus
        self._insert_milvus(dense_vector,sparse_vector,file_title,item_name)

        # 6、回填
        self._fill_item_name(state,chunks,item_name)

        return state

    def _validate_state(self, state):

        file_title = state.get('file_title')
        chunks = state.get('chunks')
        item_name_chunks_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        if not file_title:
            raise StateFieldError(node_name=self.node_name,field_name='file_title',expected_type=str,message=f'文件{file_title}:不存在')

        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(node_name=self.node_name, field_name='chunks', expected_type=list,message='chunks不存在或不是列表格式')
        if not item_name_chunks_k or item_name_chunks_k <=0:
            raise ValidationError(node_name=self.node_name,message='item_name_chunks_k为空或无效')

        if not item_name_chunk_size or item_name_chunk_size <=0:
            raise ValidationError(node_name=self.node_name, message='item_name_chunk_size为空或无效')

        return  file_title,chunks,item_name_chunks_k,item_name_chunk_size

    def _item_name_recognition_context(self, chunks:List[Dict[str,Any]], item_name_chunks_k:int, item_name_chunk_size:int)->str:
        """
        商品名识别上下文提取
        Args:
            chunks: 总体文段
            item_name_chunks_k: 最大提取块数
            item_name_chunk_size: 总切片的最大文段长度

        Returns:

        """
        total = 0
        final_context = []
        for index,chunk in enumerate(chunks[:item_name_chunks_k]):
            if not isinstance(chunk,dict):
                continue
            chunk_content = chunk.get("content")
            context = f"【切片】-{index+1}-{chunk_content}"

            if total+len(context) > item_name_chunk_size:
                break

            total +=len(context)
            final_context.append(context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_name_context:str, file_title:str)->str:
        """
        调用llm模型识别上下文，提取商品名
        Args:
            item_name_context:商品名上下文
            file_title:文档名

        Returns:item_name

        """
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"LLM客户端创建失败，降级使用文件标题{file_title}作为商品名，原因：{str(e)}")
            return file_title

        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title = file_title,context = item_name_context)

        try:
            llm_res = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]).content.strip('')

            if not llm_res or llm_res =="UNKNOWN":
                self.logger.warning(f'LLM提取商品名失败，降级使用文档标题{file_title}作为商品名')
                return file_title
            self.logger.info(f"文档：{file_title},LLM提取商品名为：{llm_res}")
            return llm_res
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败，降级使用文档标题{file_title}作为商品名,原因：{str(e)}")
            return file_title

    def _embedding_item_name(self, item_name:str | Sequence[str])-> tuple[None, None] | tuple[list[float], Any] | tuple[
        list[list[float]], list[Any]]:
        """
        主要职责：商品名向量化嵌入
        Args:
            item_name:商品名

        Returns:
            稠密向量dense_vector、稀疏向量sparse_vector

        """
        try:
            beg_m3_client :BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE_m3嵌入模型调用失败：{str(e)}")
            return None,None

        # 当为一个文档时，简单的写法
        # try:
        #     # 2. 计算稠密和稀疏向量
        #     vector_result = beg_m3_client.encode_documents(documents=[item_name])    # 返回一个二维数组即array[array[data]],整体结构为字典：{"dense":[[data]]}
        #     # 当有多个文档时返回[[data1],[data2]]
        #     # {'dense': [[-0.02663    0.00758   -0.03772    0.01645    0.002884  -0.02704,  0.002026  -0.002975   0.03014    0.01197   -0.0055    -0.001892, -0.0348     0.003767   0.02272    0.02403   -0.00863    0.01272,  0.04318   -0.03009   -0.01868   -0.006954   0.06116    0.02466, -0.0008373  0.01755    0.01551   -0.01549    0.005096  -0.03955, -0.01027   -0.0005994  0.02571   -0.05978   -0.0465    -0.03677,  0.0233    -0.01338   -0.009026  -0.001125   0.014656  -0.0133,  0.02496   -0.03558   -0.01528   -0.04916   -0.0283     0.00692, -0.01271    0.005367   0.0659    -0.004257   0.0631    -0.0341, -0.00275    0.02563    0.004013   0.03983   -0.02446   -0.0002255,  0.0004268  0.02563   -0.006702   0.002836   0.00847    0.07886, -0.02594   -0.003868   0.000967   0.02042   -0.004757   0.0122, -0.02452    0.01868   -0.05902    0.00712    0.01659   -0.010605, -0.0289     0.01546    0.05325    0.004814  -0.00439    0.02997,  0.02774    0.006367  -0.00976    0.0601    -0.011734   0.0119, -0.02368    0.004246   0.02193    0.0011835 -0.02971    0.01376, -0.02957    0.06244   -0.003832   0.004517   0.01683    0.0635, -0.009254  -0.02965    0.05765   -0.007664   0.01627    0.006256, -0.002367  -0.01201    0.046     -0.03583    0.05353    0.008934,  0.01657   -0.0194    -0.02266    0.02385    0.01971   -0.02403,  0.01039    0.06494   -0.006973  -0.02536   -0.02794   -0.01756,  0.006912   0.00985   -0.003641   0.04047    0.04712    0.01129, -0.05127   -0.0007896 -0.02458   -0.02605   -0.012794   0.002077, -0.03305   -0.0322     0.0446     0.03424   -0.009865  -0.05405,  0.02724   -0.08655    0.0164    -0.01029    0.002449  -0.007492, -0.01234    0.0328     0.006256  -0.002243  -0.02592   -0.00404, -0.01309    0.05167   -0.02136    0.03452    0.02557   -0.04324, -0.01965    0.02829    0.02487    0.00961    0.04535   -0.02638, -0.011284  -0.00961   -0.02808    0.003738  -0.03668    0.012985,  0.0386     0.02179    0.01023    0.005665   0.02739   -0.01116,  0.01456    0.012245  -0.02524   -0.0694    -0.0441     0.00735,  0.01738   -0.001228  -0.0003462  0.001262  -0.001235  -0.01422,  0.00972   -0.003836   0.0209    -0.02896   -0.004482  -0.00378, -0.003506  -0.03265    0.01478    0.03577    0.00812   -0.00514, -0.029      0.03406   -0.04526   -0.01709    0.02383   -0.01301, -0.00983    0.010864   0.014854   0.00217   -0.04062    0.00909,  0.00922    0.01935   -0.03024   -0.02184   -0.006702  -0.01762, -0.00362    0.02066    0.010765  -0.02162   -0.003096   0.03085,  0.01636    0.01726   -0.01374   -0.002478   0.000221   0.013695,  0.0385    -0.01282   -0.0121     0.00905    0.022     -0.01147, -0.01558   -0.02966   -0.00814    0.02747   -0.01675    0.000373, -0.02234   -0.03012   -0.01117    0.005615   0.01143   -0.01499, -0.03412    0.02284    0.02199    0.05896  ]],
        #     # 'sparse': <Compressed Sparse Row sparse array of dtype 'float64' with 5 stored elements and shape (1, 250002)>}
        #
        #     # 3. 解析稠密向量和稀疏向量（处理单个文档即len(documents)=1）
        #     # 3.1 获取稠密向量

        #     # vector_result["dense"] 通常是一个 NumPy 数组：执行vector_result.get("dense")[0]得到第一个文档的向量：shape == (1024,)但它仍然是 NumPy 数组：array([-0.02663, 0.00758, ...])所以需要用tolist转换为python原生列表
        #     dense_vector = vector_result.get('dense')[0].tolist()
        #     # 3.2 获取稀疏向量矩阵 csr(解开)
        #     sparse_csr = vector_result.get('sparse')
        #     # 3.1 获取行索引
        #     start_index = sparse_csr.indptr[0]
        #     end_index = sparse_csr.indptr[1]
        #     # 3.2 获取 token_id
        #     token_id = sparse_csr.indices[start_index:end_index].tolist()
        #     # 3.3 获取 weight
        #     weight = sparse_csr.data[start_index:end_index].tolist()
        #     # 3.4 构建{"token_id":weight}字典结构
        #
        #     sparse_vector = dict(zip(token_id, weight))
        #
        #     self.logger.info(f"稠密向量维度：{len(dense_vector)}")
        #     return dense_vector, sparse_vector
        # except Exception as e:
        #     self.logger.error(f"BGE_M3嵌入模型计算{item_name}失败，原因：{str(e)}")
        #     return None, None

        try:
            # 2. 计算稠密和稀疏向量
            vector_result = beg_m3_client.encode_documents(documents=[item_name])    # 返回一个二维数组即array[array[data]],整体结构为字典：{"dense":[[data]]}
            dense_vector = vector_result.get('dense')[0].tolist()
            # 3 获取稀疏向量矩阵 csr(解开)
            sparse_csr = vector_result.get('sparse')
            sparse_csr_dim = sparse_csr.shape[0]
            # 3.1 获取行索引
            for i in range(sparse_csr_dim):
                start_index = sparse_csr.indptr[i]
                end_index = sparse_csr.indptr[i+1]
                # 3.2 获取 token_id
                token_id = sparse_csr.indices[start_index:end_index].tolist()
                # 3.3 获取 weight
                weight = sparse_csr.data[start_index:end_index].tolist()
                # 3.4 构建{"token_id":weight}字典结构
                sparse_vector = dict(zip(token_id, weight))

            self.logger.info(f"稠密向量维度：{len(dense_vector)}")
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE_M3嵌入模型计算{item_name}失败，原因：{str(e)}")
            return None, None






        try:
            # 当为多个文档或者一个文档时，复杂写法
            # 1、计算稠密向量、稀疏向量
            documents = [item_name] if isinstance(item_name,str) else list(item_name)

            if not documents:
                return  None,None

            vector_result = beg_m3_client.encode_documents(documents=documents)
            # 2、提取稠密向量与稀疏向量
            # 2.1、提取稠密向量

            dense_row = vector_result['dense']
            dense_rows = (
                dense_row.tolist() if hasattr(dense_row,"tolist") else list(dense_row)
            )


            # 当为单一文档时，返回一维数组
            if dense_rows and isinstance(dense_row[0],numbers.Real):
                dense_rows = [dense_rows]

            dense_vectors = [
                [float(value) for value in row]
                for row in dense_rows
            ]

            # 2.2、获取稀疏向量矩阵csr
            sparse_raw = vector_result["sparse"]
            sparse_csr = sparse_raw.tocsr()

            sparse_vectors = []

            for row_index in range(sparse_csr.shape[0]):
                start = int(sparse_csr.indptr[row_index])
                end = int(sparse_csr.indptr[row_index + 1])

                sparse_row = {}

                for index, value in zip(
                        sparse_csr.indices[start:end],
                        sparse_csr.data[start:end],
                ):
                    index = int(index)
                    value = float(value)

                    if index >= 0 and value != 0.0 and math.isfinite(value):  #math.isfinite判断一个数值是否为有限数
                        sparse_row[index] = value

                sparse_vectors.append(sparse_row)

            if len(dense_vectors) != len(documents):
                raise ValueError("稠密向量数量与文档数量不一致")

            if len(sparse_vectors) != len(documents):
                raise ValueError("稀疏向量数量与文档数量不一致")

            if not dense_vectors or not sparse_vectors:
                self.logger.error("商品名向量提取失败")
            if len(documents)  == 1:
                dense_vector = dense_vectors[0]
                sparse_vector = sparse_vectors[0]
                self.logger.info(f"稠密向量维度：{len(dense_vectors)}")
                return dense_vector,sparse_vector
            else:
                self.logger.info(f"稠密向量维度：{len(dense_vectors)}")
                return dense_vectors,sparse_vectors
        except Exception as e:
            self.logger.error(f"BGE_M3嵌入模型计算{item_name}失败，原因：{str(e)}")
            return None,None

    def _insert_milvus(self, dense_vector:Optional[List], sparse_vector:Optional[Dict[str,Any]], file_title:str, item_name:str):
        """
        主要职责：将已向量化的识别商品名保存到 Milvus数据库中
        row 行记录{"dense_vector":"值","sparse_vector":"值","file_title":"文档","item_name":"商品名"}
        Args:
            dense_vector:稠密向量
            sparse_vector:稀疏向量
            file_title:文档名
            item_name:商品名

        Returns:

        """

        if not dense_vector or not sparse_vector:
            return

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败，原因：{str(e)}")
            return

        item_name_collection = self.config.item_name_collection

        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection,milvus_client)

            item_name_data_row = {
                "file_title":file_title,
                "item_name":item_name,
                "dense_vector":dense_vector,
                "sparse_vector":sparse_vector
            }

            inserted_result = milvus_client.insert(collection_name=item_name_collection,data=[item_name_data_row])
            self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")
        except Exception as e:
            self.logger.error(f"商品名{item_name}向量化嵌入失败,原因：{str(e)}")



    def _create_item_name_collection(self, item_name_collection:str, milvus_client:MilvusClient):
        """
        主要职责：创建商品名集合
        Args:
            item_name_collection: 集合名称
            milvus_client: 客户端

        Returns:

        """
        try:
            # 1. 创建 schema 约束
            schema = milvus_client.create_schema()
            # 创建主键约束
            schema.add_field(field_name="product",datatype=DataType.VARCHAR,max_length=65535,is_primary = True,auto_id = True)
            # 创建标量字段约束
            schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length = 65535)

            schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length = 65535)

            schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim =1024)

            schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 2. 创建索引(标量字段建立索引 向量字段建立索引)
            index_params = milvus_client.prepare_index_params()

            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="AUTOINDEX",
                metric_type ="COSINE"
            )

            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"
            )
            # 3. 创建集合
            milvus_client.create_collection(collection_name=item_name_collection,schema=schema,index_params=index_params)

            self.logger.info(f"创建{item_name_collection}集合成功")
        except Exception as e:
            self.logger.error(f"集合{item_name_collection}创建失败，原因：{str(e)}")

    def _fill_item_name(self, state:ImportGraphState, chunks:List[Dict], item_name:str):
        """
         回填 item_name:
         位置一：回填给 chunk 方便下游模型使用
         位置二：回填给 state 方便其它节点使用
         Args:
             state:
             item_name:

         Returns:
         """
        for chunk in chunks:
            chunk['item_name'] = item_name

        state["item_name"] = item_name


if __name__ == '__main__':

    setup_logging()

    # 1. 读取 chunk.json
    chunk_json_path = r"G:\AI-project\shopkeeper_brain\knowledge\test\BGE\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建 state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }
    item_name_chunk_path = r"G:\AI-project\shopkeeper_brain\knowledge\test\BGE\item_name_chunks.json"
    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用 process
    result = node.process(state)
    with open(item_name_chunk_path,'w',encoding="utf-8") as f:
        json.dump(result.get('chunks'),f,ensure_ascii=False,indent=4)


    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks 数量: {len(result.get('chunks', []))}")
    print(f"首个 chunk 是否含 item_name: {'item_name' in result['chunks'][0]}")






























