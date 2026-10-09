from FlagEmbedding import FlagReranker


def get_rerank_model():
    reranker = FlagReranker(
        model_name_or_path= r"F:\ai_models\modelscope_cache\models\BAAI\bge-reranker-large",
        devices="cuda",
        use_fp16=True
    )
    return reranker



