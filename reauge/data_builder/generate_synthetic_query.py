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

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    index_dir = cfg.data_builder.generate_synthetic_query.index_dir
    from_num_nodes = cfg.data_builder.generate_synthetic_query.from_num_nodes
    output_path = cfg.data_builder.generate_synthetic_query.output_path
    random_seed = cfg.data_builder.generate_synthetic_query.random_seed

    service_context = ServiceContext.from_defaults()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)
    random.seed(random_seed)
    nodes = sorted(index.docstore.docs.values(), key=lambda x: x.id_)               
    assert from_num_nodes <= len(nodes)
    selected_indices = random.sample(range(len(nodes)), from_num_nodes)
    selected_nodes = [nodes[idx] for idx in selected_indices]

    llm = OpenAI(model="gpt-4")

    qa_dataset = generate_question_context_pairs(
        selected_nodes, llm=llm, num_questions_per_chunk=2
    )
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    qa_dataset.save_json(output_path)

if __name__ == "__main__":
    main()
