from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from omegaconf import DictConfig, OmegaConf
import hydra
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
)
import pandas as pd


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    index_dir = cfg.retriever.evaluate.index_dir
    test_data_path = cfg.retriever.evaluate.test_data_path
    metrics = cfg.retriever.evaluate.metrics

    service_context = ServiceContext.from_defaults()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)
    retriever = index.as_retriever()
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )
    total_metrics = {m: 0.0 for m in metrics}
    qa_data = EmbeddingQAFinetuneDataset.from_json(test_data_path)
    metric_dicts = []
    for qid, query in list(qa_data.queries.items())[:3]:
        relevant_doc_ids = qa_data.relevant_docs[qid]
        result = retriever_evaluator.evaluate(
            query=query, expected_ids=relevant_doc_ids
        )
        metric_dicts.append(result.metric_vals_dict)
    full_df = pd.DataFrame(metric_dicts)
    for metric in metrics:
        metric_ave_val = full_df[metric].mean()
        print(f"{metric}: {metric_ave_val}")


if __name__ == "__main__":
    main()
