#import streamlit as st
from llama_index.query_engine.citation_query_engine import (
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
from llama_index import ServiceContext
from llama_index.response_synthesizers import get_response_synthesizer, ResponseMode

from llama_index.prompts import PromptTemplate
from llama_index.schema import MetadataMode
from llama_index.query_engine import CitationQueryEngine


#@st.cache_resource
def init_query_engine(
    index_dir,
    _llm,
    _citation_cfg,
    enable_node_expander=False,
    streaming=True,
    semantic_scholar=False,
):

    synthesizer_service_context = ServiceContext.from_defaults(llm=_llm)
    citation_qa_template = CITATION_QA_TEMPLATE

    if semantic_scholar:
        retriever = SemanticScholarRetriever(topk=_citation_cfg.similarity_top_k)
        node_postprocessors = None
        query_engine_callback_manager = synthesizer_service_context.callback_manager

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
        query_engine_callback_manager = index.service_context.callback_manager
    
    if _citation_cfg.citation_qa_template_path:
        with open(
            _citation_cfg.citation_qa_template_path, "r", encoding="utf-8"
        ) as f:
            citation_qa_template = PromptTemplate(f.read())

    response_synthesizer = get_response_synthesizer(
        service_context=synthesizer_service_context,
        text_qa_template=citation_qa_template,
        refine_template=CITATION_REFINE_TEMPLATE,
        response_mode=ResponseMode.COMPACT,
        use_async=False,
        streaming=streaming,
    )

    # service_context for the synthesizer is same as service_context of the index
    query_engine = CitationQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        callback_manager=query_engine_callback_manager,
        citation_chunk_size=_citation_cfg.citation_chunk_size,
        node_postprocessors=node_postprocessors,
        metadata_mode=MetadataMode.LLM,
    )

    return query_engine


def replace_with_identifiers(s):
    # Initialize a counter
    mapping = {}

    # Define a function to use as replacement in re.sub
    def replacement(match):
        matched_str = int(match.group().lstrip("[").rstrip("]"))
        if matched_str not in mapping:
            mapping[matched_str] = len(mapping) + 1

        return f"[{mapping[matched_str]}]"

    # Define the regex pattern
    pattern = r"\[\d+\]"
    # Replace all matches of the pattern with the new identifiers
    new_s = re.sub(pattern, replacement, s)
    return new_s, mapping
