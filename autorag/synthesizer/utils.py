# import streamlit as st
from llama_index.core.query_engine.citation_query_engine import (
    CITATION_QA_TEMPLATE,
    CITATION_REFINE_TEMPLATE,
)
import re
from autorag.indexer.expanded_indexer import ExpandedIndexer
from autorag.retriever.google_and_vector_retriever import (
    GoogleAndVectorRetriever,
    GoogleRetriever,
)
from autorag.retriever.semantic_scholar_retriever import SemanticScholarRetriever
from llama_index.core import Settings
from llama_index.core.response_synthesizers import CompactAndRefine

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode
from llama_index.core.query_engine import CitationQueryEngine


# @st.cache_resource
def init_query_engine(
    index_dir,
    _llm,
    _citation_cfg,
    enable_node_expander=False,
    streaming=True,
    semantic_scholar=False,
):

    # Set global settings
    Settings.llm = _llm

    citation_qa_template = CITATION_QA_TEMPLATE

    if semantic_scholar:
        retriever = SemanticScholarRetriever(topk=_citation_cfg.similarity_top_k)
        node_postprocessors = None
        query_engine_callback_manager = Settings.callback_manager

    else:
        expanded_index = ExpandedIndexer.load(index_dir, enable_node_expander)
        index = expanded_index.index
        retriever = index.as_retriever(similarity_top_k=_citation_cfg.similarity_top_k)
        if _citation_cfg.google_search_topk > 0:
            google_retriever = GoogleRetriever(topk=_citation_cfg.google_search_topk)
            retriever = GoogleAndVectorRetriever(retriever, google_retriever)

        node_postprocessors = (
            [expanded_index.node_expander] if enable_node_expander else None
        )
        query_engine_callback_manager = Settings.callback_manager

    if _citation_cfg.citation_qa_template_path:
        with open(_citation_cfg.citation_qa_template_path, "r", encoding="utf-8") as f:
            citation_qa_template = PromptTemplate(f.read())

    response_synthesizer = CompactAndRefine(
        llm=_llm,
        text_qa_template=citation_qa_template,
        refine_template=CITATION_REFINE_TEMPLATE,
        streaming=streaming,
    )

    query_engine = CitationQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        callback_manager=query_engine_callback_manager,
        citation_chunk_size=_citation_cfg.citation_chunk_size,
        node_postprocessors=node_postprocessors,
        metadata_mode=MetadataMode.LLM,
    )

    return query_engine


def replace_with_identifiers(s, existing_mapping=None):
    # Initialize a counter
    mapping = existing_mapping if existing_mapping is not None else {}
    next_identifier = max(mapping.values(), default=0) + 1

    # Define a function to use as replacement in re.sub
    def replacement(match):
        nonlocal next_identifier
        matched_str = int(match.group().lstrip("[").rstrip("]"))
        if matched_str not in mapping:
            mapping[matched_str] = next_identifier
            next_identifier += 1

        return f"[{mapping[matched_str]}]"

    # Define the regex pattern
    pattern = r"\[\d+\]"
    # Replace all matches of the pattern with the new identifiers
    new_s = re.sub(pattern, replacement, s)

    # Filter the mapping to include only identifiers mentioned in the current string
    current_identifiers = set(
        int(m.group().lstrip("[").rstrip("]")) for m in re.finditer(pattern, s)
    )
    filtered_mapping = {k: v for k, v in mapping.items() if k in current_identifiers}

    return new_s, filtered_mapping
