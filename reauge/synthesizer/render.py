import streamlit as st
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    index_dir = cfg.synthesizer.render.index_dir
    app_description = cfg.synthesizer.render.app_description
    
    service_context = ServiceContext.from_defaults()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)

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
            message_placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message) # Add response to message history

if __name__ == "__main__":
    main()
