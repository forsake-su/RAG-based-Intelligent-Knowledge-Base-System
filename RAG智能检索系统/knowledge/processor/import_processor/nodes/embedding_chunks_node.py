from pathlib import Path
from typing import Any,List,Dict
import json


# 项目包
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError, StateFieldError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients,BGEM3EmbeddingFunction



class EmbeddingChunksNode(BaseNode):
    node_name = "embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:


        self.log_step("【Step1】","校验参数")
        chunks = self._validate_state(state)

        self.log_step("【Step2】", "向量化嵌入")

        self._embed_insert(chunks)


        return state


    def _validate_state(self, state:ImportGraphState):

        chunks = state.get('chunks')

        if not chunks or not isinstance(chunks,List):
            raise StateFieldError(node_name=self.node_name,field_name='chunks',expected_type=List)

        for i,chunk in enumerate(chunks) :
            if not isinstance(chunk,dict):
                raise ValidationError(f"【chunk_{i}】类型无效，实际类型：{type(chunk).__name__}")

        return chunks

    def _embed_insert(self, chunks:List[Dict[str,Any]]):

        try:
            embedding_model = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"BGE_M3 嵌入模型调用失败，原因：{str(e)}")
            raise EmbeddingError(message="BGE_M3 嵌入模型调用失败",node_name=self.node_name)

        total = len(chunks)
        batch_size = self.config.embedding_batch_size

        if not batch_size:
            self.logger.error(f"batch_size值为空")
            raise ValueError(f"batch_size值为空")

        embed_result = []
        for i in range(0,total,batch_size):
            batch = chunks[i:i+batch_size]
            self.logger.info(f"嵌入批次【{i+1}-{i+len(batch)}】/{total}")

            result = self._embed_chunks(batch,embedding_model)
            embed_result.extend(result)
        return embed_result

    def _embed_chunks(self, batch:List[Dict[str,Any]], embedding_model:BGEM3EmbeddingFunction)->List[Dict[str,Any]]:
        """
        主要职责：将每批次chunk进行向量化，并将结果注入到chunks中
        Args:
            batch: 每批次chunk
            embedding_model: BGEM3EmbeddingFunction

        Returns:

        """
        text = [f"{chunk.get("item_name","")}\n{chunk.get('content','')}" for chunk in batch]

        try:
            vector_result = embedding_model.encode_documents(text)
        except Exception as e:
            raise EmbeddingError(message=f"文段{text}：向量化失败",node_name=self.node_name)

        if not vector_result:
            raise EmbeddingError(f"BGE-M3编码结果为空",node_name=self.node_name)

        sparse_csr = vector_result['sparse']
        for i,chunk in enumerate(batch):
            chunk['dense_vector'] = vector_result.get('dense')[i].tolist()
            chunk['sparse_vector'] = self._extract_sparse_vector(sparse_csr,i)

        return batch

    def _extract_sparse_vector(self, sparse_csr, i:int)->Dict[str,Any]:
        start_index = sparse_csr.indptr[i]
        end_index = sparse_csr.indptr[i+1]

        token_id = sparse_csr.indices[start_index:end_index].tolist()
        weight = sparse_csr.data[start_index:end_index].tolist()

        return dict(zip(token_id,weight))


if __name__ =="__main__":
    setup_logging()

    base_dir = Path(r"G:\AI-project\shopkeeper_brain\knowledge\test\BGE")

    input_path = base_dir/"item_name_chunks.json"
    output_path = base_dir/"chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path,"r",encoding="utf-8") as f:
        chunks_data = json.load(f).get('chunks')

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data})

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result_state,f,ensure_ascii=False,indent=4)
    print(f"向量生成完成，结果已保存至:\n{output_path}")










