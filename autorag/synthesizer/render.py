import streamlit as st
from llama_index import ServiceContext, PromptHelper
from omegaconf import DictConfig, OmegaConf
import hydra
from llama_index.llms import OpenAI
from llama_index.response_synthesizers import get_response_synthesizer, ResponseMode

from llama_index.query_engine import CitationQueryEngine
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)
from llama_index.schema import MetadataMode
from llama_index.prompts import PromptTemplate
from llama_index.query_engine.citation_query_engine import (
    CITATION_QA_TEMPLATE,
    CITATION_REFINE_TEMPLATE,
)
import re
from autorag.indexer.expanded_indexer import ExpandedIndexer


# Create an instance of the GlobalHydra class
global_hydra = hydra.core.global_hydra.GlobalHydra()

# Call the clear() method on the instance
global_hydra.clear()


def init(index_dir, openai_model_name, citation_cfg, enable_node_expander=False):

    expanded_index = ExpandedIndexer.load(index_dir, enable_node_expander)

    index = expanded_index.index
    node_postprocessors = (
        [expanded_index.node_expander] if enable_node_expander else None
    )

    llm = OpenAI(model=openai_model_name, temperature=0)
    synthesizer_service_context = ServiceContext.from_defaults(llm=llm)

    if citation_cfg.citation_qa_template_path:
        with open(citation_cfg.citation_qa_template_path, "r", encoding="utf-8") as f:
            citation_qa_template = PromptTemplate(f.read())
    else:
        citation_qa_template = CITATION_QA_TEMPLATE

    response_synthesizer = get_response_synthesizer(
        service_context=synthesizer_service_context,
        text_qa_template=citation_qa_template,
        refine_template=CITATION_REFINE_TEMPLATE,
        response_mode=ResponseMode.COMPACT,
        use_async=False,
        streaming=True,
    )

    # service_context for the synthesizer is same as service_context of the index
    query_engine = CitationQueryEngine.from_args(
        index,
        response_synthesizer=response_synthesizer,
        similarity_top_k=citation_cfg.similarity_top_k,
        # here we can control how granular citation sources are, the default is 512
        citation_chunk_size=citation_cfg.citation_chunk_size,
        node_postprocessors=node_postprocessors,
        metadata_mode=MetadataMode.LLM,
    )

    return query_engine


def show_feedback_component(message_id):
    # if the component is already rendered on the webpage, do nothing
    if message_id in [
        feedback["message_id"] for feedback in st.session_state.feedbacks
    ]:
        return

    cols = st.columns([0.1, 1])
    with cols[0]:
        if st.button("👍", key=f"thumbs_up_{message_id}"):
            st.write("thanks!")
            st.session_state.feedbacks.append(
                {"message_id": message_id, "is_good": True, "feedback": ""}
            )
    with cols[1]:
        if st.button("👎", key=f"thumbs_down_{message_id}"):
            reason = st.text_input(
                "Please let us know why this response was not helpful",
                key=f"reason_{message_id}",
            )
            if reason:
                st.session_state.feedbacks.append(
                    {"message_id": message_id, "is_good": False, "feedback": reason}
                )
                st.write("thanks!")


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


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_cfg = cfg.synthesizer.render
    index_dir = cur_cfg.index_dir
    app_description = cur_cfg.app_description
    citation_cfg = cur_cfg.citation_cfg
    enable_hyde = cur_cfg.enable_hyde
    enable_node_expander = cur_cfg.enable_node_expander
    openai_model_name = cur_cfg.openai_model_name
    show_retrieved_nodes = cur_cfg.show_retrieved_nodes

    query_engine = init(
        index_dir,
        openai_model_name,
        citation_cfg,
        enable_node_expander,
    )
    if enable_hyde:
        hyde = HyDEQueryTransform(include_original=True)

    st.header("Chat with Your Documents (only support single-turn conversation now)")

    if "messages" not in st.session_state.keys():  # Initialize the chat message history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Ask me a question about {app_description}!",
            }
        ]

    if "feedbacks" not in st.session_state.keys():
        st.session_state.feedbacks = []

    if prompt := st.chat_input(
        "Your question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message_id, message in enumerate(
        st.session_state.messages
    ):  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

        # show feedback component when the message is sent by the assistant
        if message["role"] == "assistant":
            show_feedback_component(message_id)

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if enable_hyde:
                spinner_msg = "Generating hypothetical response"
                with st.spinner(spinner_msg):
                    prompt = hyde(prompt)
                    full_response = f"=== Non-RAG response ===\n\n{prompt.embedding_strs[0]}\n\n=== RAG response ===\n\n"

            raw_rag_response = ""
            response = query_engine.query(prompt)
            for ans in response.response_gen:
                raw_rag_response += ans
                rag_response, mapping = replace_with_identifiers(raw_rag_response)
                message_placeholder.markdown(full_response + rag_response + "▌")
            full_response += rag_response

            full_response += "\n\n### References\n\n"
            for raw_ref_id, new_ref_id in mapping.items():
                ref_node = response.source_nodes[raw_ref_id - 1]
                full_response += (
                    f"#### [{new_ref_id}]\n\n"
                    + "\n\n"
                    + ref_node.node.get_text()
                    + "\n\n"
                )

            message_placeholder.markdown(full_response)

            if show_retrieved_nodes:
                with st.expander("Retrieved nodes"):
                    retrieved_node_info = ""
                    for idx, retrieved_node in enumerate(response.source_nodes):
                        retrieved_node_info += (
                            f"#### retrieved node [{idx + 1}]\n\n"
                            + "\n\n```"
                            + retrieved_node.node.get_text()
                            + "```\n\n"
                        )
                    st.write(retrieved_node_info)

        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)  # Add response to message history

        # Show feedback components to make sure it is displayed after the message is fully returned
        show_feedback_component(len(st.session_state.messages) - 1)


if __name__ == "__main__":
    main()
