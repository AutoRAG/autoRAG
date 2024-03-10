import streamlit as st
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from sqlalchemy import (
    create_engine,
    MetaData
)
from llama_index.core.query_engine import NLSQLTableQueryEngine

engine = create_engine("sqlite:///airbnb_ny.db")
metadata_obj = MetaData()
metadata_obj.reflect(engine)


llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

sql_database = SQLDatabase(engine, metadata=metadata_obj, include_tables=["airbnb_ny"])

#print(response)
# "Find accommodation near the train station."
# st.set_page_config(page_title="text 2 sql", page_icon="🤖", layout="wide")

# st.header("Turn text into sql query")

# # Create a text input field
# user_input = st.text_input("Enter some text")

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["airbnb_ny"],
    llm=llm
)

query_str = "Find accommodation near the train station."
response = query_engine.query(query_str)
print(response)
print(response.metadata['sql_query'])
print(response.metadata['result'])
#st.write(f"The processed text is: {response}")
