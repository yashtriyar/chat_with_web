import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#for google gemini api

# from langchain_google_genai import ChatGoogleGenerativeAI  
# from langchain.vectorstores import FAISS 
# from langchain.chains.question_answering import VectorDBQA


load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))

    return vector_store

#---------------------------------------------------------------------------------------------------------------
#using gemini pro

# def get_context_retriever_chain(vector_store):
#     llm = ChatGoogleGenerativeAI(model="gemini-pro") 
#     retriever = FAISS.from_documents(vector_store.as_list(), vector_store.config)

#     # Simplified prompt due to Gemini's limitations
#     # prompt = """Given the conversation history (chat_history), provide relevant information:""" 
#     prompt = ChatPromptTemplate.from_messages([
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#       ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
#     ]) 
    
#     retriever_chain = VectorDBQA.from_llm_and_retriever(llm, retriever, prompt)
#     return retriever_chain

#-----------------------------------------------------------------------------------------------------------

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")
st.sidebar.markdown("Created by: Yash Triyar ❤️‍🔥")


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

elif openai_api_key is None or openai_api_key == "":
        st.info("Please enter your OpenAI API key.")
    

else:
# main content
    #OPENAI_API_KEY=openaikey
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hey Buddy, I am your bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # user input
    user_query = st.chat_input("Lets get started...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
