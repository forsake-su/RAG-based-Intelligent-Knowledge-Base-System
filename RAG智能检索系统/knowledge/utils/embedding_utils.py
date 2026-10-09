from typing import List, Dict, Any
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

def generate_bge_m3_hybrid_vector(model:BGEM3EmbeddingFunction,embedding_documents:List[str])->Dict[str,Any]:
    """
    BGE_M3混合向量生成工具（稠密+稀疏）
    Args:
        model: BGE_M3嵌入模型
        embedding_documents:要生成嵌入向量的文本列表

    Returns:{"dense_vector":[],"sparse_vector":[]}
    """

    # 1、参数校验
    if not embedding_documents or not isinstance(embedding_documents,list):
        raise ValueError("embedding_documents文件校验失败：为空或为非列表")

    if not all(isinstance(doc,str) and doc.strip() for doc in embedding_documents):
        raise ValueError("embedding_documents 中存在无效元素（空字符串或非字符串类型）")

    # 2、生成嵌入向量

    try:
        embedding_result = model.encode_documents(embedding_documents)
    except Exception as e:
        raise RuntimeError(f"BGE-M3嵌入生成失败，原因：{str(e)}")

    # 3、校验结果
    if "dense" not in embedding_result or "sparse" not in embedding_result:
        raise ValueError(f"嵌入结果缺少必要字段，实际返回：{list(embedding_result.keys())}")

    # 4、提取稠密向量和稀疏向量

    sparse_vector = []
    dense_vector = []
    sparse_csr = embedding_result["sparse"]

    try:
        for i in range(len(embedding_documents)):
            start = sparse_csr.indptr[i]
            end = sparse_csr.indptr[i+1]
            token_ids = sparse_csr.indices[start:end].tolist()
            weights = sparse_csr.data[start:end].tolist()
            sparse_vector.append(dict(zip(token_ids,weights)))

            dense_vector.append(embedding_result["dense"][i].tolist())
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"向量解析失败，原因: {str(e)}")


    # 5、返回

    return {
        "dense_vector":dense_vector,
        "sparse_vector":sparse_vector
    }

