from dotenv import load_dotenv
load_dotenv()


from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated



class ChatState(BaseModel):
    messages:Annotated[list,add_messages]



llm=ChatGroq(model="openai/gpt-oss-20b")





def chatBotNode(state:ChatState)-> ChatState:
    response=llm.invoke(state.messages)
    state.messages=[response]
    return state


memory=InMemorySaver()




graph=StateGraph(ChatState)
graph.add_node("chatbot",chatBotNode)
graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)



graph=graph.compile(checkpointer=memory)




# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())


while True:
    query=input("User....")
    if(query).lower() in ["bye","exit","quit"]:
        print("Thanks for using mee")
        break
    response=graph.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":1}}
        )
    ans=response["messages"][-1].content
    print("AI : ",ans)