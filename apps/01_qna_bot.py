from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st



llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("AskBuddy - AI QNA Bot")

st.markdown("My QNA bot is powered by Google Gemini and langchain.You can ask any question and get an answer from the AI.")


if "messages" not in st.session_state:
    st.session_state.messages =  []



for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask a question: ")

if query:

    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    result=llm.invoke(query)
    st.chat_message("ai").markdown(result.content[0]["text"])
    st.session_state.messages.append({"role":"ai","content":result.content[0]["text"]})


# while True:
#     query=input("Ask a question: ")

#     if query.lower() in ["exit","quit","bye"]:
#         print("Good Bye....")
#         break

#     result=llm.invoke(query)
#     print("AI:",result.content[0]["text"],"\n\n")