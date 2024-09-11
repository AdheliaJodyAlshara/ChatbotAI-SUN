import tempfile
import shutil
from typing import List
from langchain.agents import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
# from typing import Annotated

from config import data_input, llm

# def searchFromInternet(query: str):
#     retriever = TavilySearchAPIRetriever(k=10, search_depth="advanced")
#     result = retriever.invoke(query)
#     return result


# def searchFromInternetUsingGoogle(query: str):
#     # https://python.langchain.com/docs/integrations/tools/google_serper
#     search = GoogleSerperAPIWrapper(type="news")
#     results = search.results(query)
#     return results

def default_tools():
    answer = """
    """
    return answer

# def chart_generator(user_question : Annotated[str, "The original user question without paraphrasing for creating chart"]) -> str:
def chart_generator(user_question : str) -> str:
    df = SmartDataframe(data_input, config={"llm": llm})
    chart_path = df.chat(user_question)

    # file_ = open(chart_path, "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()

    response = f"""The chart has been generated.
    
    Please strictly add this HTML command in your response to show the chart image to the user:
    <img src="{chart_path}" alt="chart image">"""
    return response


def initialize_tools() -> List[Tool]:
    search_tavily = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search_tavily)

    tools = [
        Tool(
            name="searchFromInternet",
            func=tavily_tool.run,
            description="useful to get additional information from internet."
        ),
        Tool(
            name="ChartGenerator",
            func=chart_generator,
            description="useful to create chart based on user question. Please address the entire user question directly without any paraphrasing."
        )
    ]

    return tools