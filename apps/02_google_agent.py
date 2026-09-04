from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool


llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# llm = ChatGroq(model="openai/gpt-oss-120b")
search=GoogleSerperAPIWrapper()



agent=create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="you are a agent and can search for any question on google",
    checkpointer=InMemorySaver()
)


while True:
    query=input("Enter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit' or query.lower() == 'quit':
        print("Good byeeeeee....!")
        break


    response=agent.invoke(
    {"messages":[{"role":"user","content":query}]},
    {"configurable": {"thread_id": "history_save"}}
    )


    print(response["messages"][-1].content[0]["text"].replace("**", ""))


    



