import streamlit as st
import pandas as pd
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from custom_tools import initialize_tools
from langchain_openai import ChatOpenAI

# Load data
data_input = pd.read_csv('YTD_sales_report_2024.csv')

# Initialize tools
tools = initialize_tools()

# Define the prefix and suffix for the prompt
prefix = '''You are the best Sales Report Analyst in Indonesia represent SUN Terra Company (https://sunterra.id) that can help me, a Data Analyst, in summarizing findings and drawing insights from a sales report data.
You should not make things up and only answer all questions related to renewable energy and solar panel.

SUNTerra Company Background:
PT Energi Indonesia Berkarya (SUN terra) is a leading residential, social, and commercial solar panel company in Indonesia. SUN terra is a business line of SUN Energy, the largest solar energy project developer in Indonesia, with solar projects exceeding 280 MWp since its establishment in 2016. Focusing on the development of solar panels for residential areas, we carry the mission to increase the use of solar panels as an environmentally friendly alternative energy source for everyone through the use of application-based technology. SUN Terra Company focuses on Residential Solar Panel Installation.

Lead Across Sales Conversion Journey:
- L0 = New Leads
- L1 = In Approach
- L2 = Initial Quotation Sent (indicative proposal)
- L3 = Term sheet signed by customer
- L4 = Negotiation with customer
- L5 = Confirmation Fee/Deposit
- L6 = Contract signed. User pays confirmation fee/subs/deposit.

Here the steps for you to summarize and give insight about the SUN Terra sales report data:
- Step 1 : Step by step analyze L0 until L6 trends within years over months from each lead type. You don't need to use the tools for this step.
- Step 2 : From step 1 get additional information from the internet to get insight and enhance your summarization. REMEMBER YOU ARE ONLY PERMITTED TO SEARCH FROM THE INTERNET 3 TIMES OR LESS! If you feel enough with your research from the internet less than 3 times, you can immediately move on to the next step.
- Step 3 : From step 2, provide the final answer after repeating the search for responses from the internet only 3 times with a summary and insight based on data and information from the internet.

If the question is a follow-up question or does not relate to the provided sales data, then here the steps for you:
- Step 1 : Get the information from the internet to get answer from the user question. REMEMBER YOU ARE ONLY PERMITTED TO SEARCH FROM THE INTERNET 3 TIMES OR LESS! If you feel enough with your research from the internet less than 3 times, you can immediately move on to the next step.
- Step 2 : From step 1, provide the final answer.
'''

data_string = f'''Sales Report Data of Sun Terra Business Unit: 
```
{data_input}
```

'''

suffix = data_string + '''Your past conversation with human:
```
{chat_history}
```

Begin!

Question: {human_input}
Thought: {agent_scratchpad}
'''

# Define the format instructions for the agent's output
format_instructions = """Use the following format without any punctuations:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, MUST be one of these tool names only without the parameters [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated only 3 times)
Thought: Now I know the final answer
Final Answer: If the question directly involves analyzing the sales report data provided. Final Answer Format:

<Lead_type>
Summarization: <Your Summarization as a paragraph>
Insight: <Your Insight as a paragraph>


If the question is a follow-up or does not relate to the provided sales data:
Final Answer: <Directly provide the summarized answer without the detailed format>
"""

# Create the prompt for the ZeroShotAgent
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    format_instructions=format_instructions,
    input_variables=["chat_history", "human_input", "agent_scratchpad"],
)

memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the language model chain with a chat model
llm_chain = LLMChain(
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-4o"
    ),
    prompt=prompt
)

# Create the ZeroShotAgent with the language model chain
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

# Initialize the AgentExecutor with the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)

if __name__ == "__main__":
    # Setup the Streamlit interface
    st.title('Q&A AI SUN Terra Sales Data')

    # Initialize session state for maintaining conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

    # User inputs their question
    user_question = st.chat_input("Enter your question about the sales data...")

    # Button to process the question
    if user_question:
        # Append user's question to the session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Run the agent with the formulated prompt
        with st.spinner('Processing...'):
            response = agent_executor.run(human_input=user_question)
        
        # Append AI response to the session state
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
