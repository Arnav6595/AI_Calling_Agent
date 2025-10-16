import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

df = pd.DataFrame({
    'combined_text': ['test text'],
    'col1': ['value1'],
    'col2': ['value2']
})

try:
    loader = DataFrameLoader(df, page_content_column='combined_text', metadata_columns=['col1', 'col2'])
    docs = loader.load()
    print("Success: metadata_columns is supported.")
    print(docs[0].metadata)
except TypeError as e:
    print(f"Error: {e}")