import os
import hydra
import json
import re
import uuid
from typing import Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from tqdm import tqdm
from llama_index.schema import MetadataMode, TextNode
from llama_index.evaluation import (
    EmbeddingQAFinetuneDataset,
)
from thefuzz import fuzz


# generate queries as a convenience function
def generate_qa_pairs_from_annotated_pairs(
    nodes: List[TextNode],
    annotated_pairs: List[Tuple],
    fuzzy_match_score_threshold: float = 70,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}
    for question, annotated_text in tqdm(annotated_pairs):
        best_matched_node_score = max(
            [(node, fuzz.partial_ratio(annotated_text, node.text)) for node in nodes],
            key=lambda x: x[1],
        )
        if best_matched_node_score[1] >= fuzzy_match_score_threshold:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [best_matched_node_score[0].node_id]

    # construct dataset
    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    index_dir = cfg.data_builder.build_from_annotated_retrieval_data.index_dir
    annotated_data_path = (
        cfg.data_builder.build_from_annotated_retrieval_data.annotated_data_path
    )
    output_path = cfg.data_builder.build_from_annotated_retrieval_data.output_path
    question_field = cfg.data_builder.build_from_annotated_retrieval_data.question_field
    annotated_field = (
        cfg.data_builder.build_from_annotated_retrieval_data.annotated_field
    )

    df = pd.read_excel(annotated_data_path, header=0)

    service_context = ServiceContext.from_defaults()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)

    nodes = index.docstore.docs.values()

    annotated_pairs = zip(df[question_field], df[annotated_field])
    qa_data_from_annotated = generate_qa_pairs_from_annotated_pairs(
        nodes, annotated_pairs
    )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    qa_data_from_annotated.save_json(output_path)


if __name__ == "__main__":
    main()
