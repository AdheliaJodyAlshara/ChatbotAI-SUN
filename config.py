import os
import pandas as pd
from langchain_openai import ChatOpenAI

url = os.getenv("CSV_URL")
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
data_input = pd.read_csv(dwn_url)
data_input_str = data_input.to_string()

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    max_tokens=4096,
    seed=42
)