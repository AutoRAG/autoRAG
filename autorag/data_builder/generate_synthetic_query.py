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

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_cfg = cfg.data_builder.generate_synthetic_query
    index_dir = cur_cfg.index_dir
    from_num_nodes = cur_cfg.from_num_nodes
    output_path = cur_cfg.output_path
    random_seed = cur_cfg.random_seed
    prompt_template_path = cur_cfg.prompt_template_path
    num_questions_per_chunk = cur_cfg.num_questions_per_chunk

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
    if prompt_template_path:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            qa_generate_prompt_tmpl = f.read().strip('\n')
    else:
        qa_generate_prompt_tmpl = DEFAULT_QA_GENERATE_PROMPT_TMPL
    print(qa_generate_prompt_tmpl)
    qa_dataset = generate_question_context_pairs(
        selected_nodes, llm=llm, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl, num_questions_per_chunk=num_questions_per_chunk
    )
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    qa_dataset.save_json(output_path)

if __name__ == "__main__":
    main()
