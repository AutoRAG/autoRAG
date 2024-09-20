import streamlit as st
from llama_index.llms import OpenAI
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv

from llama_index.indices.query.query_transform import HyDEQueryTransform
from autorag.synthesizer.utils import init_query_engine, replace_with_identifiers
from llama_index.chat_engine.condense_question import (
    DEFAULT_PROMPT as DEFAULT_CONDENSE_PROMPT,
)
from hydra.core.global_hydra import GlobalHydra
from llama_index.schema import MetadataMode

if GlobalHydra.instance().is_initialized():
    # Clear the current Hydra instance if it is initialized
    GlobalHydra.instance().clear()


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
    enable_node_expander = cur_cfg.enable_node_expander
    openai_model_name = cur_cfg.openai_model_name
    show_retrieved_nodes = cur_cfg.show_retrieved_nodes
    reference_url = cur_cfg.reference_url
    include_historical_messages = cur_cfg.include_historical_messages
    streaming = True

    llm = OpenAI(model=openai_model_name, temperature=0)
    query_engine_med_qa = init_query_engine(
        index_dir,
        llm,
        citation_cfg,
        enable_node_expander,
        streaming,
    )

    query_engine_semantic_scholar = init_query_engine(
        index_dir,
        llm,
        citation_cfg,
        enable_node_expander,
        streaming,
        semantic_scholar=True,
    )
    st.header(f"{app_description.upper()} Chatbot Demo")

    semantic_scholar = st.toggle("Use Semantic Scholar")
    if semantic_scholar:
        query_engine = query_engine_semantic_scholar
    else:
        query_engine = query_engine_med_qa

    if enable_hyde and not semantic_scholar:
        hyde = HyDEQueryTransform(include_original=True)

    if "messages" not in st.session_state.keys():  # Initialize the chat message history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Hello, how can I help you!",
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
        # if message["role"] == "assistant":
        #    show_feedback_component(message_id)

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        # condense historical message into one question
        if include_historical_messages and len(st.session_state.messages) > 2:
            condense_prompt_template = DEFAULT_CONDENSE_PROMPT
            chat_history_str = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]]
            )
            prompt = llm.predict(
                condense_prompt_template, question=prompt, chat_history=chat_history_str
            )
            st.write(f"(rewritten query)\n\n{prompt}")
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if enable_hyde and not semantic_scholar:
                spinner_msg = "Generating hypothetical response"
                with st.spinner(spinner_msg):
                    prompt = hyde(prompt)
                    full_response = f"=== Raw GPT ({openai_model_name}) response ===\n\n{prompt.embedding_strs[0]}\n\n=== AutoRAG response ===\n\n"

            raw_rag_response = ""
            print(prompt)
            response = query_engine.query(prompt)
            for ans in response.response_gen:
                raw_rag_response += ans
                rag_response, mapping = replace_with_identifiers(raw_rag_response)
                message_placeholder.markdown(full_response + rag_response + "▌")
            full_response += rag_response

            if mapping:
                full_response += "\n\n### References\n\n"
            for raw_ref_id, new_ref_id in mapping.items():
                ref_node = response.source_nodes[raw_ref_id - 1]
                if reference_url:
                    url = ref_node.metadata["url"].strip("\n")
                    full_response += f"[{new_ref_id}] [{url}]({url})\n\n"
                else:
                    full_response += (
                        f"#### [{new_ref_id}]\n\n"
                        + "\n\n"
                        + ref_node.node.get_content(metadata_mode=MetadataMode.LLM)
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

        def reset_conversation():
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"Hello, how can I help you!",
                }
            ]
            st.rerun()

        st.button("Clear conversation", on_click=reset_conversation)
        # Show feedback components to make sure it is displayed after the message is fully returned
        # show_feedback_component(len(st.session_state.messages) - 1)


if __name__ == "__main__":
    load_dotenv()
    main()
