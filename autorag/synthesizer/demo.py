import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the API endpoints
DEVICE_PORT = os.getenv("DEVICE_PORT", "3000")
SCHOLAR_PORT = os.getenv("SCHOLAR_PORT", "3001")
DRUG_PORT = os.getenv("DRUG_PORT", "3002")

DEVICE_API_URL = f"http://127.0.0.1:{DEVICE_PORT}/query"
SCHOLAR_API_URL = f"http://127.0.0.1:{SCHOLAR_PORT}/query"
DRUG_API_URL = f"http://127.0.0.1:{DRUG_PORT}/query"

st.header("AutoRAG Chatbot Demo")

data_source = st.selectbox("Select Data Source", ["DEVICE", "DRUG", "SCHOLAR"])


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how can I help you!"}
    ]

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message_id, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Prepare the request data

        # Choose the appropriate API URL based on the selection
        if data_source == "DEVICE":
            api_url = DEVICE_API_URL
        elif data_source == "DRUG":
            api_url = DRUG_API_URL
        elif data_source == "SCHOLAR":
            api_url = SCHOLAR_API_URL
        else:
            raise ValueError("Invalid data source")

        request_data = {
            "prompt": prompt,
            "include_historical_messages": True,
            "chat_history": st.session_state.messages[:-1],
            "use_semantic_scholar": data_source == "SCHOLAR",
        }

        # Make the API call
        response = requests.post(api_url, json=request_data)
        if response.status_code == 200:
            data = response.json()
            rag_response = data["response"]
            references = data["references"]
            source_nodes = data["source_nodes"]

            full_response = rag_response

            if references:
                full_response += "\n\n### References\n\n"
                for ref in references:
                    metadata_str = json.dumps(ref["metadata"], indent=2)
                    url = ref["metadata"].get("url", None)
                    if ref["metadata"].get("document_type", None) == "webpage":
                        full_response += f"#### [{ref['id']}]\n\n{metadata_str}\n\n"
                    else:
                        full_response += f"#### [{ref['id']}]\n\n{metadata_str}\n\n{ref['content']}\n\n"

            message_placeholder.markdown(full_response)

            with st.expander("Retrieved nodes"):
                for idx, node_text in enumerate(source_nodes):
                    st.write(
                        f"#### Retrieved node [{idx + 1}]\n\n```{node_text}```\n\n"
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            st.error(
                f"Failed to get response from the API. Status code: {response.status_code}"
            )


def reset_conversation():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how can I help you!"}
    ]
    st.rerun()


st.button("Clear conversation", on_click=reset_conversation)

if __name__ == "__main__":
    pass
