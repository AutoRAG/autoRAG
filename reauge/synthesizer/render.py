import streamlit as st
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from omegaconf import DictConfig, OmegaConf
import hydra
from llama_index.query_engine import CitationQueryEngine

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_cfg = cfg.synthesizer.render
    index_dir = cur_cfg.index_dir
    app_description = cur_cfg.app_description
    citation_cfg = cur_cfg.citation_cfg

    service_context = ServiceContext.from_defaults()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)
    if citation_cfg.enable_cite:
        # service_context for the synthesizer is same as service_context of the index
        query_engine = CitationQueryEngine.from_args(
            index,
            similarity_top_k=citation_cfg.similarity_top_k,
            # here we can control how granular citation sources are, the default is 512
            citation_chunk_size=citation_cfg.citation_chunk_size,
            streaming=True,
        )
    else:
        query_engine = index.as_query_engine(service_context=service_context, streaming=True)

    st.header("Chat with Your Documents (only support single-turn conversation now)")

    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ask me a question about {app_description}!"}
        ]

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = query_engine.query(prompt)
            for ans in response.response_gen:
                full_response += ans
                message_placeholder.markdown(full_response + "▌")
            if citation_cfg.enable_cite:
                full_response += '\n\n### References\n\n'
                for idx, ref in enumerate(response.source_nodes):
                    full_response += f"[{idx+1}]\n\n" + "```\n\n" + ref.node.get_text() + "\n\n```\n\n"
            message_placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message) # Add response to message history

if __name__ == "__main__":
    main()
