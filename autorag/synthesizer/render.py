import streamlit as st
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from omegaconf import DictConfig, OmegaConf
import hydra
from llama_index.query_engine import CitationQueryEngine
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

# Create an instance of the GlobalHydra class
global_hydra = hydra.core.global_hydra.GlobalHydra()

# Call the clear() method on the instance
global_hydra.clear()


# initialize the index
@st.cache_data
def init(index_dir, similarity_top_k, citation_chunk_size, enable_cite):
    service_context = ServiceContext.from_defaults()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)
    if enable_cite:
        # service_context for the synthesizer is same as service_context of the index
        query_engine = CitationQueryEngine.from_args(
            index,
            similarity_top_k=similarity_top_k,
            # here we can control how granular citation sources are, the default is 512
            citation_chunk_size=citation_chunk_size,
            streaming=True,
        )
    else:
        query_engine = index.as_query_engine(
            service_context=service_context, streaming=True
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


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_cfg = cfg.synthesizer.render
    index_dir = cur_cfg.index_dir
    app_description = cur_cfg.app_description
    citation_cfg = cur_cfg.citation_cfg
    enable_hyde = cur_cfg.enable_hyde

    query_engine = init(
        index_dir,
        citation_cfg.similarity_top_k,
        citation_cfg.citation_chunk_size,
        citation_cfg.enable_cite,
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
            response = query_engine.query(prompt)
            for ans in response.response_gen:
                full_response += ans
                message_placeholder.markdown(full_response + "▌")
            if citation_cfg.enable_cite:
                full_response += "\n\n### References\n\n"
                for idx, ref in enumerate(response.source_nodes):
                    full_response += (
                        f"[{idx+1}]\n\n"
                        + "```\n\n"
                        + ref.node.get_text()
                        + "\n\n```\n\n"
                    )
            message_placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)  # Add response to message history

        # Show feedback components to make sure it is displayed after the message is fully returned
        show_feedback_component(len(st.session_state.messages) - 1)


if __name__ == "__main__":
    main()
