from dotenv import load_dotenv

load_dotenv()


from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st




## data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []



def process_document(path):
    ## load the documents

    # loader = PyPDFLoader("../data/medical_report.pdf")
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()


    ## split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_text = splitter.split_documents(documents=docs)


    ## embedding and vector DB
    # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


    vector_db = InMemoryVectorStore.from_documents(
        documents=split_text, embedding=embeddings
    )


    ## create a Agent - tool , llm , prompt

    llm = ChatGroq(model="openai/gpt-oss-20b")


    @tool
    def retrive_context(query: str):
        """
        Retrieve documents relevent to a query from the knowledge base
        """
        context = ""

        docs = vector_db.similarity_search(query=query, k=3)
        for doc in docs:
            context = doc.page_content + "\n\n"

        return context


    system_prompt = """
    you are a helpful assistant that answers questions using retrived context
    my knowlege  base consits of the details from the uploaded documents.
    always use the 'retrive_context' tool for questions requiring external knowlege.
    """


    memory = InMemorySaver()


    agent = create_agent(
        model=llm,
        tools=[retrive_context], 
        system_prompt=system_prompt, 
        checkpointer=memory
    )

    st.session_state.agent=agent
    st.session_state.document_uploaded=True


# while True:
#     query = input("USER : ")
#     if query.lower() == "quit":
#         break

#     response = agent.invoke(
#         {"messages": [{"role": "user", "content": query}]},
#         {"configurable": {"thread_id": 1}},
#     )
#     result = response["messages"][-1].content

#     print("AI : ", result)



# upload UI

if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(label="Select PDF Files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./docs_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()


# chat UI
if st.session_state.document_uploaded and st.session_state.agent:

    for message in st.session_state.messages:
        role=message.get("role")
        content=message.get("content")
        st.chat_message(role).markdown(content)
  


    query=st.chat_input("Ask Anything related to uploaded document....")


    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response=st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable": {"thread_id": 1}}

        )


        answer=response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})




















































# --------------------------------------------------------------------------------------------------------
# THIS IS WOKRING CODE BUT NOT ADD WEB

# from dotenv import load_dotenv

# load_dotenv()


# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import InMemoryVectorStore
# from langchain.agents import create_agent
# from langchain.tools import tool
# from langgraph.checkpoint.memory import InMemorySaver
# import streamlit as st


# ## load the documents

# loader = PyPDFLoader("../data/medical_report.pdf")
# docs = loader.load()


# ## split into chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# split_text = splitter.split_documents(documents=docs)


# ## embedding and vector DB
# # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


# vector_db = InMemoryVectorStore.from_documents(
#     documents=split_text, embedding=embeddings
# )


# ## create a Agent - tool , llm , prompt

# llm = ChatGroq(model="openai/gpt-oss-20b")


# @tool
# def retrive_context(query: str):
#     """
#     Retrieve documents relevent to a query from the knowledge base
#     """
#     context = ""

#     docs = vector_db.similarity_search(query=query, k=3)
#     for doc in docs:
#         context = doc.page_content + "\n\n"

#     return context


# system_prompt = """
# you are a helpful assistant that answers questions using retrived context
# my knowlege  base consits of the details from the uploaded documents.
# always use the 'retrive_context' tool for questions requiring external knowlege.
# """


# memory = InMemorySaver()


# agent = create_agent(
#     model=llm, tools=[retrive_context], system_prompt=system_prompt, checkpointer=memory
# )


# while True:
#     query = input("USER : ")
#     if query.lower() == "quit":
#         break

#     response = agent.invoke(
#         {"messages": [{"role": "user", "content": query}]},
#         {"configurable": {"thread_id": 1}},
#     )
#     result = response["messages"][-1].content

#     print("AI : ", result)
