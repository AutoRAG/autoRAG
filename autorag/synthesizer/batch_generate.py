"""
Generate and save answers given queries from a excel file
"""

import json
import hydra
from omegaconf import DictConfig
import pandas as pd
import time
import os

from llama_index.indices.query.query_transform import HyDEQueryTransform
from autorag.synthesizer.utils import init_query_engine, replace_with_identifiers
from autorag.data_builder.generate_synthetic_query import QUERY_NAME_FIELD


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_cfg = cfg.synthesizer.eval
    index_dir = cur_cfg.index_dir
    citation_cfg = cur_cfg.citation_cfg
    enable_hyde = cur_cfg.enable_hyde
    enable_node_expander = cur_cfg.enable_node_expander
    openai_model_name = cur_cfg.openai_model_name
    excel_input_path = cur_cfg.excel_input_path
    output_dir = cur_cfg.output_dir
    max_num_queries = cur_cfg.max_num_queries or -1
    start_query_idx = cur_cfg.start_query_idx or 0
    query_field_name = cur_cfg.query_field_name or QUERY_NAME_FIELD
    streaming = False

    query_engine = init_query_engine(
        index_dir,
        openai_model_name,
        citation_cfg,
        enable_node_expander,
        streaming,
    )
    if enable_hyde:
        hyde = HyDEQueryTransform(include_original=True)

    df = pd.read_excel(excel_input_path, header=0)

    os.makedirs(output_dir, exist_ok=True)
    for idx, ori_query in enumerate(df[query_field_name]):
        if idx < start_query_idx:
            continue
        if idx == start_query_idx + max_num_queries:
            break
        print(idx, ori_query)
        query = ori_query
        if enable_hyde:
            query = hyde(query)
        ans = query_engine.query(query)
        response, mapping = replace_with_identifiers(ans.response)

        reference = ""
        for raw_ref_id, new_ref_id in mapping.items():
            ref_node = ans.source_nodes[raw_ref_id - 1]
            reference += (
                f"#### [{new_ref_id}]\n\n" + "\n\n" + ref_node.node.get_text() + "\n\n"
            )
        save_list = {"query": ori_query, "answer": response, "reference": reference}
        output_path = os.path.join(output_dir, f"query_{idx}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(save_list))
        # Avoid per min limit
        time.sleep(60)


if __name__ == "__main__":
    main()
