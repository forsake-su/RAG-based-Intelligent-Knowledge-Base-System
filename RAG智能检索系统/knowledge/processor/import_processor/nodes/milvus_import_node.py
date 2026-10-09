# 内置包
from pathlib import Path
from typing import Any,List,Dict,Tuple,Sequence,Optional
import json
import logging
from dataclasses import dataclass

#三方包
from pymilvus import MilvusClient,DataType
from pymilvus.orm.schema import CollectionSchema



# 项目包
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.exceptions import ValidationError, StateFieldError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name:str
    datatype:DataType
    max_length:Optional[int] = None

_SCALAR_FIELD:Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="cur_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)  #Sequence[ScalarFieldSpec]  类型注解，表明是ScalarFieldSpec的对象，且是一个有序的、可通过下标（索引）访问的只读的对象集合



class _MilvusSchemaBuilder:

    @staticmethod
    def build(milvus_client : MilvusClient , dim : int)->CollectionSchema:
        logger.info(f"【Step1】: 构建向量参数schema")
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="chunk_id",datatype=DataType.INT64,is_primary = True,auto_id = True)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,dim = dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        logger.info(f"【Step2】: 构建标量参数schema")

        for spec in _SCALAR_FIELD:
            kwargs:Dict[str,Any] = {
                "field_name":spec.field_name,
                "datatype":spec.datatype,
                "max_length":spec.max_length if spec.max_length is not None else None
            }
            schema.add_field(**kwargs)
        logger.info(f"Schema构建完成")
        return schema



class _MilvusIndexBuilder:
    @staticmethod
    def build(milvus_client : MilvusClient , collection_name : str):
        logger.info(f"开始构建集合 {collection_name} 索引...")

        index = milvus_client.prepare_index_params(collection_name=collection_name)
        index.add_index(field_name="dense_vector",index_name="dense_vector_index",index_type="AUTOINDEX",metric_type = "COSINE")
        index.add_index(field_name="sparse_vector",index_name="sparse_vector_index",index_type="SPARSE_INVERTED_INDEX",metric_type = "IP")
        logger.info(f"集合 {collection_name} 索引构建完成")
        return index

class _MilvusInserter:

    def __init__(self, milvus_client : MilvusClient , collection_name:str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert(self, chunks:List[Dict[str,Any]])->List[Dict[str,Any]]:
        logger.info(f"开始插入 {len(chunks)} 条记录到 Milvus...")
        result = self._milvus_client.insert(collection_name=self._collection_name,data=chunks)
        ids = result.get("ids",[])
        insert_count = result.get("insert_count",0)

        for chunk,chunk_id in zip(chunks,ids):
            chunk["chunk_id"] = chunk_id
        logger.info(f"插入完成：{insert_count} 条记录，已回填 chunk_id")
        return chunks


class MilvusImportNode(BaseNode):

    node_name = "milvus_import_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.log_step("【Step1】","参数校验")
        validate_chunks,dim=self._validate_state(state)

        self.log_step("【Step2】", "创建集合")
        milvus_client,collection_name= self._create_collection(dim)

        self.log_step("【Step3】", "参数入库")
        inserter = _MilvusInserter(milvus_client = milvus_client,collection_name =collection_name )

        self.log_step("【Step4】", "更新状态")
        state['chunks'] = inserter.insert(chunks =validate_chunks )

        return  state


    def _validate_state(self, state:ImportGraphState):

        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(node_name=self.node_name,field_name="chunks",message=f"chunks为空或类型无效,实际类型：{type(chunks).__name__}")

        validated_chunks = []
        for i,chunk in enumerate(chunks):
            if not isinstance(chunk,dict):
                raise ValidationError(message=f"chunk{i}类型无效，实际为：{type(chunk).__name__}")

            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunk[{i}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError(message="所有 chunk 均无有效向量，无法入库",node_name=self.node_name)

        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效chunk数量：{len(validated_chunks)},向量维度：{dim}")

        return validated_chunks,dim

    def _create_collection(self,dim):

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            raise MilvusError(message=f"Milvus客户段创建失败，原因：{str(e)}")

        collection_name = self.config.chunks_collection

        if milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"Milvus中已存在集合{collection_name}")
            return milvus_client,collection_name

        schema = _MilvusSchemaBuilder.build(milvus_client,dim)
        index_params = _MilvusIndexBuilder.build(milvus_client,collection_name)

        milvus_client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)
        self.logger.info(f"集合：{collection_name}创建完成")
        return milvus_client,collection_name



def _cli_main() -> None:

    import  json
    setup_logging()

    temp_dir = Path(r"G:\AI-project\shopkeeper_brain\knowledge\test\BGE"
    )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get("chunks")}

    node = MilvusImportNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")

if __name__ =="__main__":
    _cli_main()












