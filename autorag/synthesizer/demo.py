import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the API endpoints
RAG_PORT = os.getenv("RAG_PORT", "3002")
SCHOLAR_PORT = os.getenv("SCHOLAR_PORT", "3003")

RAG_API_URL = f"http://127.0.0.1:{RAG_PORT}/query"
SCHOLAR_API_URL = f"http://127.0.0.1:{SCHOLAR_PORT}/query"

st.header("AutoRAG Chatbot Demo")

data_source = st.selectbox("Select Data Source", ["RAG", "SCHOLAR"])


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
        if data_source == "RAG":
            api_url = RAG_API_URL
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

        # Make the API call and stream the response
        with requests.post(api_url, json=request_data, stream=True) as response:
            if response.status_code == 200:
                full_response = ""
                all_references = []  # Collect all references here
                for line in response.iter_lines():
                    if line:
                        # Decode the line from bytes to string
                        line = line.decode("utf-8")
                        data = json.loads(line)
                        rag_response = data["response"]
                        references = data["references"]

                        # Update the response text in real-time
                        full_response += rag_response
                        message_placeholder.markdown(full_response)

                        # Collect references for later
                        all_references.extend(references)

                # After the response is complete, append references
                if all_references:
                    references_text = "\n\n### References\n\n"
                    for ref in all_references:
                        metadata_str = json.dumps(ref["metadata"], indent=2)
                        url = ref["metadata"].get("url", None)
                        if ref["metadata"].get("document_type", None) == "webpage":
                            references_text += (
                                f"#### [{ref['id']}]\n\n{metadata_str}\n\n"
                            )
                        else:
                            references_text += f"#### [{ref['id']}]\n\n{metadata_str}\n\n{ref['content']}\n\n"

                    # Append references to the full response
                    full_response += references_text
                    message_placeholder.markdown(full_response)

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
