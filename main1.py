import streamlit as st
import pandas as pd
import os
import re
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
from custom_tools import initialize_tools, data_input
from callbacks import StreamHandler, stream_data
from config import llm

# Set the page layout to wide
st.set_page_config(layout="wide")

# url = os.getenv("CSV_URL")
# file_id=url.split('/')[-2]
# dwn_url='https://drive.google.com/uc?id=' + file_id
# data_input = pd.read_csv(dwn_url)


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
- Step 1 : Step by step analyze provided sales report data L0 until L6 trends within years over months from each lead type. You don't need to use the tools for this step.
- Step 2 : Enhance your analysis from Step 1 by gathering additional insights from the internet to strengthen your summary. You are limited to a maximum of three internet searches. If you find the information you need before reaching three searches, you can proceed to the next step without completing all three searches. But if the question involves creating a chart, you can use the Chart Generator tool by passing the user question without paraphrasing for creating chart.
- Step 3 : Summarize the findings and provide insights based on your analysis from Step 1 and the additional information from Step 2. Ensure your final answer integrates the data trends with insights from the internet searches or chart generator.
- Step 4 : In the final output, You should include all reference data & links to back up your research; You should include all reference data.

If the question is a follow-up question or does not relate to the provided sales report data, then here the steps for you:
- Step 1 : Get the information from the internet to get answer from the user question. REMEMBER YOU ARE ONLY PERMITTED TO SEARCH FROM THE INTERNET 3 TIMES OR LESS! If you feel enough with your research from the internet less than 3 times, you can immediately move on to the next step.
- Step 2 : From step 1, provide the final answer. In the final output, You should include all reference data & links to back up your research; You should include all reference data.
'''

data_string = f'''Sales Report Data of Sun Terra Business Unit: 
```
{data_input.to_string()}
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
format_instructions = """Strictly use the following format and it must be in consecutive order without any punctuation:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, MUST be one of these tool names only without the parameters [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated only 3 times)
Thought: I now know the final answer
Final Answer: If the question directly involves analyzing the sales report data provided. 

Provide your final answer using the following output format for each lead type:
<Lead Type>
- Summarization: <Your Summarization as a paragraph> 
- Insight: <Your Insight as a paragraph>


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

# stream_handler = StreamHandler(st.empty())

# Initialize the language model chain with a chat model
llm_chain = LLMChain(
    llm=llm,
    #     ChatOpenAI(
    #     model_name="gpt-4o",
    #     temperature=0
    #     # streaming=True,
    #     # callbacks=[stream_handler]
    # ),
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
    st.title('Q&A AI')

    # Initialize session state for maintaining conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "<img" in message['content']:
                image_path = re.search(r'src="([^"]*)"', message['content']).group(1)
                new_message = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', message['content'])
                st.markdown(new_message)
                st.image(image_path)
            else:
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
            try:
                response = agent_executor.run(human_input=user_question)
            except:
                response = "I'm sorry I can't process your query right now. Please try again."

        # Append AI response to the session state
        try:
            with st.chat_message("assistant"):
                if "<img" in response:
                    image_path = re.search(r'src="([^"]*)"', response).group(1)
                    new_response = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', response)
                    st.write_stream(stream_data(new_response))
                    st.image(image_path)
                else:
                    st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
        except:
            with st.chat_message("assistant"):
                response = "I'm sorry I can't process your query right now. Please try again."
                st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
