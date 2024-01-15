# reauge
Reauge is a package for easy development for retrieval augmented generation system using command lines.

## Install
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
python -m reauge.indexer.build ++app_name=<your_app_name>
```
### Prepare a test dataset
Given some annotated data (question, reference) pairs, you can use the following command to prepare test data.
```
python -m reauge.data_builder.build_from_annotated_retrieval_data ++app_name=<your_app_name> 
```
If you don't have annotated data, you can use the following command to generate some synthetic data from LLM. Note that this step may be costly if you use OPENAI API. Use a small `data_builder.generate_synthetic_query.from_num_nodes` value in `conf/config.yaml` to avoid high cost.
```
python -m reauge.data_builder.generate_synthetic_query ++app_name=<your_app_name>
```

### Evaluate the retriever
```
python -m reauge.retriever.evaluate ++app_name=<your_app_name>
```
### Run a chatbot app
Note that you need to be in the entry directory of this repo to run the chatbot. 
```
streamlit run reauge/synthesizer/render.py ++app_name=<your_app_name>
```

## TODO
1. Add evaluation for synthesizer
2. Optimize the code
3. Add unit test
4. Add advanced PDF preprocessor
