from llama_index.llms import OpenAI
from omegaconf import DictConfig
import websockets
import json
import asyncio
import hydra

from llama_index.indices.query.query_transform import HyDEQueryTransform
from autorag.synthesizer.utils import ws_query_engine, replace_with_identifiers
from llama_index.chat_engine.condense_question import (
    DEFAULT_PROMPT as DEFAULT_CONDENSE_PROMPT,
)
from hydra.core.global_hydra import GlobalHydra
from llama_index.schema import MetadataMode

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

session_state = {
    "messages": [],
    "feedbacks": []
}

async def handle_message(websocket, path, cfg: DictConfig):
    cur_cfg = cfg.synthesizer.websocket_server
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
    query_engine = ws_query_engine(
        index_dir,
        llm,
        citation_cfg,
        enable_node_expander,
        streaming,
    )
    if enable_hyde:
        hyde = HyDEQueryTransform(include_original=True)

    async for prompt in websocket:

        if "messages" not in session_state.keys():  # Initialize the chat message history
            session_state['messages'] = [
                {
                    "role": "assistant",
                    "content": f"Hello, how can I help you!",
                }
            ]

        session_state['messages'].append({"role": "user", "content": prompt})

        full_response = ""

        if include_historical_messages and len(session_state['messages']) > 2:
            condense_prompt_template = DEFAULT_CONDENSE_PROMPT
            chat_history_str = "\n".join(
                [f"{m['role']}: {m['content']}" for m in session_state['messages'][:-1]]
            )
            prompt = llm.predict(
                condense_prompt_template, question=prompt, chat_history=chat_history_str
            )
            response = {
                "response": f"Received your message: {prompt}"
            }
            await websocket.send(json.dumps(prompt))

        if enable_hyde:
            spinner_msg = "Generating hypothetical response"
            print(f"loading message: {spinner_msg}")
            prompt = hyde(prompt)
            full_response = f"=== Raw GPT ({openai_model_name}) response ===\n\n{prompt.embedding_strs[0]}\n\n=== AutoRAG response ===\n\n"

        raw_rag_response = ""

        response = query_engine.query(prompt)
        for ans in response.response_gen:
            raw_rag_response += ans
            rag_response, mapping = replace_with_identifiers(raw_rag_response)

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

        if show_retrieved_nodes:
            retrieved_node_info = ""
            for idx, retrieved_node in enumerate(response.source_nodes):
                retrieved_node_info += (
                    f"#### retrieved node [{idx + 1}]\n\n"
                    + "\n\n```"
                    + retrieved_node.node.get_text()
                    + "```\n\n"
                )
            full_response += retrieved_node_info

        response = {
            "role": "assistant",
            "content": {full_response}
        }

        session_state['messages'].append(response)

        await websocket.send(json.dumps(response))

async def start_server(config):
    server = await websockets.serve(
        lambda ws, 
        path: handle_message(ws, path, config), 
        config.server.host, 
        config.server.port
    )
    print(f"WebSocket server started on ws://{config.server.host}:{config.server.port}")
    await server.wait_closed()

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    asyncio.run(start_server(cfg))


if __name__ == "__main__":
    main()
