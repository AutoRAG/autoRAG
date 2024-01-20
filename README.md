# AutoRAG

The AutoRAG Toolkit is specifically designed to streamline the creation and refinement of Retrieval Augmented Generation (RAG) systems. RAG, a key methodology in integrating Large Language Models (LLMs) with tailored data sets, forms an essential base for numerous LLM-driven applications. However, the complexities involved in developing RAG, notably in areas like performance evaluation and enhancement, are substantial. 
This toolkit aims to ease and accelerate the development of high-quality RAG systems through automatic evaluation-driven optimization of various system components such as data preprocessing, document indexing and retrieval, and prompting engineering.

## Installation
Prerequisite:
```
Python 3.10.9
```

In the main repo folder, run
```
pip install -e .
```

## Usage
This repo uses [Hydra](https://hydra.cc/docs/intro/) to manage the configs. Please check conf/config.yaml for the default values and use Hydra arguments to modify the default values. 

### setup OPENAI API key if you are using OPENAI API for LLM and embedding model
```
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

### Build an index from corpus
Upload your documents where the answers can be retrieved from to `data/<your_app_name>/corpus` folder.  
```
python -m autorag.indexer.build ++app_name=<your_app_name>
```
### Run a chatbot app
Note that you need to be in the entry directory of this repo to run the chatbot. 
```
streamlit run autorag/synthesizer/render.py ++app_name=<your_app_name>
```

## Evaluation
### Prepare a test dataset
Given some annotated data (question, reference) pairs, you can use the following command to prepare test data.
```
python -m autorag.data_builder.build_from_annotated_retrieval_data ++app_name=<your_app_name> 
```
If you don't have annotated data, you can use the following command to generate some synthetic data from LLM. Note that this step may be costly if you use OPENAI API. Use a small `data_builder.generate_synthetic_query.from_num_nodes` value in `conf/config.yaml` to avoid high cost.
```
python -m autorag.data_builder.generate_synthetic_query ++app_name=<your_app_name>
```

### Evaluate the retriever
```
python -m autorag.retriever.evaluate ++app_name=<your_app_name>
```

# Roadmap
## High-level features to be supported
AutoRAG targets to offer a suite of features designed to ease and accelerate the development of RAG systems:

- **Performance Evaluation and Optimization**: AutoRAG includes tools for automated and manual performance evaluation and automate the optimization process for the components in the RAG system, helping you to maximize the efficiency and accuracy of your system.

- **Automated Data Integration**: Seamlessly integrate your custom data sets with LLMs. AutoRAG streamlines the process of data retrieval and formatting.

- **Modular Design**: AutoRAG's modular architecture allows for easy customization and extension. Whether you're building a simple Q&A system or a complex conversational AI, AutoRAG can be tailored to meet your specific needs.

- **Scalable Infrastructure**: Designed with scalability in mind, AutoRAG can handle projects of any size, from small-scale experiments to large-scale deployments.

- **User-Friendly Interface**: With an intuitive interface and comprehensive documentation, AutoRAG is accessible to developers of all skill levels.

## Low-level features to be supported
- [ ] Support unstructured data
- [ ] Support structured data
- [ ] Enable component-level performance evaluation
    - [ ] Evaluation of synthesizer by checking with the question and retrieved docs
    - [ ] Evaluation of synthesizer by comparing with a ground truth answer or another answer from baseline
    - [ ] Evaluation of retrieval when there is no ground truth doc
- [ ] Enable various methods to improve performance
    - [ ] Support meta data
    - [ ] Support Azure BI Document intelligence
    - [ ] Enable chunk size variation
    - [ ] Enable Context enrichment (index-small-retrieve-big)
    - [ ] Enable summarized chunks
    - [ ] Enable HyDE
    - [ ] Enable Query rewrite
    - [ ] Enable Hybrid retrieval
    - [ ] Enable Fine-tuning retrieval model
    - [ ] Enable re-ranking
- [ ] Human feedback collection
- [ ] Automatic optimization
    - [ ] Grid search
    - [ ] Smart search (research topic)
