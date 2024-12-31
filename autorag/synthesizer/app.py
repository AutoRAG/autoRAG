from flask import Flask, request, jsonify
from llama_index.llms import OpenAI
from llama_index.indices.query.query_transform import HyDEQueryTransform
from autorag.synthesizer.utils import init_query_engine, replace_with_identifiers
from llama_index.chat_engine.condense_question import (
    DEFAULT_PROMPT as DEFAULT_CONDENSE_PROMPT,
)
from llama_index.schema import MetadataMode
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from flask_cors import CORS
import urllib
import json

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize global variables
query_engine = None
llm = None
hyde = None
port = None  # Add port as a global variable
document_bucket_name = None
app_name = None


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def init_app(cfg: DictConfig):
    global query_engine, llm, hyde, port, document_bucket_name, app_name

    cur_cfg = cfg.synthesizer.app
    index_dir = cur_cfg.index_dir
    citation_cfg = cur_cfg.citation_cfg
    enable_hyde = cur_cfg.enable_hyde
    enable_node_expander = cur_cfg.enable_node_expander
    openai_model_name = cur_cfg.openai_model_name
    document_bucket_name = cur_cfg.document_bucket_name
    print(f"document_bucket_name: {document_bucket_name}")
    streaming = True
    port = cur_cfg.port  # Set the global port variable

    llm = OpenAI(model=openai_model_name, temperature=0)

    # Initialize query engine based on app_name
    app_name = cfg.app_name
    if app_name == "scholar":
        query_engine = init_query_engine(
            index_dir,
            llm,
            citation_cfg,
            enable_node_expander,
            streaming,
            semantic_scholar=True,
        )
    else:
        query_engine = init_query_engine(
            index_dir, llm, citation_cfg, enable_node_expander, streaming
        )
        if enable_hyde:
            hyde = HyDEQueryTransform(include_original=True)

    print(f"Initialized {app_name} API")
    print(f"Port in init_app: {port}")


def main():
    global port  # Declare port as global in main
    init_app()  # Call init_app to initialize everything
    print(f"Port in main: {port}")
    app.run(host="0.0.0.0", port=port, debug=True)


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    prompt = data["prompt"]
    include_historical_messages = data.get("include_historical_messages", False)
    chat_history = data.get("chat_history", [])

    if include_historical_messages and len(chat_history) > 1:
        condense_prompt_template = DEFAULT_CONDENSE_PROMPT
        chat_history_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in chat_history]
        )
        prompt = llm.predict(
            condense_prompt_template, question=prompt, chat_history=chat_history_str
        )

    if hyde:
        prompt = hyde(prompt)

    response = query_engine.query(prompt)
    mapping = {}
    references = []

    def stream_response():
        with app.app_context():
            mapping = {}
            buffer = ""  # Buffer to store items without spaces

            def word_generator():
                nonlocal buffer
                for item in response.response_gen:
                    # If buffer exists and current item has no spaces, concatenate
                    if not buffer.strip() and " " not in item:
                        buffer += item
                        continue
                    # If we have a buffer, process it first
                    if buffer:
                        if " " not in item:
                            buffer += item
                            continue
                        # Process buffer + current item
                        full_text = buffer + item
                        buffer = ""
                    else:
                        full_text = item

                    # Split by space and yield each word
                    words = full_text.split(" ")
                    for word in words[:-1]:
                        if word:  # Only yield non-empty words
                            yield word + " "
                    if words[-1]:  # Handle the last word
                        buffer = words[-1]  # Store the last word in buffer
                if buffer:
                    yield buffer

            all_references = []
            all_ref_ids = set()
            # Use the word generator in the main loop
            for item in word_generator():
                references = []
                new_item, new_mapping = replace_with_identifiers(
                    item, existing_mapping=mapping
                )
                mapping.update(new_mapping)
                # Check for new references
                for raw_ref_id, new_ref_id in new_mapping.items():
                    ref_node = response.source_nodes[raw_ref_id - 1]
                    metadata = ref_node.node.metadata
                    if (
                        "document_name" in metadata
                        and metadata["document_name"] is not None
                        and metadata.get("url", None) is None
                    ):
                        if metadata["document_name"].endswith(".json"):
                            document_name = metadata["document_name"].replace(
                                ".json", ".pdf"
                            )
                        elif metadata["document_name"].endswith(".pdf"):
                            document_name = metadata["document_name"]
                        else:
                            document_name = metadata["document_name"] + ".pdf"
                        url_encoded_document_name = urllib.parse.quote_plus(
                            document_name
                        )
                        metadata["url"] = (
                            f"https://{document_bucket_name}.s3.amazonaws.com/{app_name}/{url_encoded_document_name}"
                        )
                    new_ref = {
                        "id": new_ref_id,
                        "content": ref_node.node.get_content(
                            metadata_mode=MetadataMode.NONE
                        ),
                        "metadata": ref_node.node.metadata,
                    }
                    references.append(new_ref)
                    if new_ref_id not in all_ref_ids:
                        all_ref_ids.add(new_ref_id)
                        all_references.append(new_ref)

                # Convert the response to JSON and then to bytes
                yield (
                    json.dumps({"response": new_item, "references": references}) + "\n"
                ).encode("utf-8")

            if len(all_references) == 0 and app_name == "scholar":
                yield (
                    json.dumps(
                        {
                            "response": "Here are some potentially relevant references.",
                            "references": all_references,
                        }
                    )
                    + "\n"
                ).encode("utf-8")

    return app.response_class(stream_response(), mimetype="application/json")


if __name__ == "__main__":
    main()
