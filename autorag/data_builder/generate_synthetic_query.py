import os
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from omegaconf import DictConfig, OmegaConf
import hydra
from llama_index.evaluation import RetrieverEvaluator
from llama_index.llms import OpenAI
import random
from llama_index.evaluation import (
    generate_question_context_pairs,
)
from llama_index.finetuning.embeddings.common import DEFAULT_QA_GENERATE_PROMPT_TMPL
from autorag.indexer.expanded_indexer import ExpandedIndexer
import pandas as pd


def save_embedding_qa_finetune_dataset_to_excel(
    qa_data, node_dict, metadata_field, output_path
):
    query_doc_list = []

    for query, doc_ids in qa_data.query_docid_pairs:
        query_doc_list.append(
            {
                "query": query,
                metadata_field: node_dict[doc_ids[0]].metadata[metadata_field],
                "doc": qa_data.corpus[doc_ids[0]],
            }
        )
    full_df = pd.DataFrame(query_doc_list)

    full_df.to_excel(output_path)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_cfg = cfg.data_builder.generate_synthetic_query
    index_dir = cur_cfg.index_dir
    from_parent_node = cur_cfg.from_parent_node
    from_num_sources = cur_cfg.from_num_sources
    random_seed = cur_cfg.random_seed
    prompt_template_path = cur_cfg.prompt_template_path
    num_questions_per_chunk = cur_cfg.num_questions_per_chunk
    openai_model_name = cur_cfg.openai_model_name
    json_output_path = cur_cfg.json_output_path
    excel_output_path = cur_cfg.excel_output_path
    metadata_field_to_save = cur_cfg.metadata_field_to_save

    expanded_index = ExpandedIndexer.load(index_dir, from_parent_node)

    if from_parent_node:
        print(
            "WARNING: using parent nodes in node expander instead of the original nodes in index_dir"
        )
        node_dict = expanded_index.node_expander.all_parent_nodes
    else:
        node_dict = expanded_index.index.docstore.docs

    sources = list(node_dict.values())

    random.seed(random_seed)

    assert from_num_sources <= len(sources)
    selected_indices = random.sample(
        range(len(sources)), min(len(sources), from_num_sources)
    )

    selected_sources = [sources[idx] for idx in selected_indices]

    llm = OpenAI(model=openai_model_name)
    if prompt_template_path:
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            qa_generate_prompt_tmpl = f.read().strip("\n")
    else:
        qa_generate_prompt_tmpl = DEFAULT_QA_GENERATE_PROMPT_TMPL

    qa_data = generate_question_context_pairs(
        selected_sources,
        llm=llm,
        qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
        num_questions_per_chunk=num_questions_per_chunk,
    )
    if json_output_path:
        output_dir = os.path.dirname(json_output_path)
        os.makedirs(output_dir, exist_ok=True)
        qa_data.save_json(json_output_path)
    if excel_output_path:
        output_dir = os.path.dirname(excel_output_path)
        os.makedirs(output_dir, exist_ok=True)
        save_embedding_qa_finetune_dataset_to_excel(
            qa_data, node_dict, metadata_field_to_save, excel_output_path
        )


if __name__ == "__main__":
    main()
