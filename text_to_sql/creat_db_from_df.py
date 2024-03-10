import numpy as np
import pandas as pd

from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert
)

df_test = pd.read_csv('airbnb_ny_filter.csv', delimiter=',', encoding='utf-8',low_memory=False)
# print(df_test.shape)

# for col in df_test.columns:
#     print(col, df_test[col].apply(type).unique())


#df_test =  df_test.head(5)
engine = create_engine("sqlite:///airbnb_ny.db")
metadata_obj = MetaData()

columns = df_test.columns
# create city SQL table
table_name = "airbnb_ny"

table_columns = []

for col in columns:
    if col == 'id':
        table_columns.append(Column(col, String(200), primary_key=True))
    elif col in ['summary', 'description', 'host_about']:
        table_columns.append(Column(col, String(2500)))
    else:
        if df_test[col].dtype == float:
            table_columns.append(Column(col, Float, nullable=True))
        else:
            table_columns.append(Column(col, String(200)))

aribnb_ny_table = Table(table_name, metadata_obj, *table_columns)
metadata_obj.create_all(engine)

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

sql_database = SQLDatabase(engine, include_tables=["airbnb_ny"])
error = []
rows = [df_test.iloc[i].to_dict() for i in range(df_test.shape[0])]
for row in rows:
    stmt = insert(aribnb_ny_table).values(**row)
    with engine.begin() as connection:
        try:
            cursor = connection.execute(stmt)
        except Exception as e:
            error.append(e)
            #print("An error occurred:", e)
           # print("The problematic row is:", row)
if len(error)==0:
    print("==================rows inserted==================")

