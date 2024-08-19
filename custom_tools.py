from langchain.agents import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

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

def initialize_tools():
    search_tavily = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search_tavily)

    tools = [
        Tool(
            name="searchFromInternet",
            func=tavily_tool.run,
            description="useful to get additional information from internet."
        )
    ]

    return tools
