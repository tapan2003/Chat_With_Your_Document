import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import Gemini
from llama_index.embeddings import GeminiEmbedding
from llama_index import SimpleDirectoryReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with Document", page_icon="📚")


# Function to load and index documents
@st.cache_resource(show_spinner=False)
def load_data():
    try:
        with st.spinner(text="Loading and indexing the Documents - hang tight! This should take 1-2 minutes."):
            # Initialize Gemini language model and embedding
            llm = Gemini(api_key="Your_Gemini_API_Key")
            embed_model = GeminiEmbedding(
                api_key="Your_Gemini_API_Key")

            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=llm, embed_model=embed_model)

            # Read documents from directory
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            documents = reader.load_data()

            # Create vector store index from documents
            index = VectorStoreIndex.from_documents(
                documents=documents, service_context=service_context)
            return index
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None


# Function to initialize chat engines
def initialize_chat_engines():
    try:
        if "chat_engine" not in st.session_state.keys():
            index = load_data()
            if index:
                st.session_state.chat_engine = index.as_chat_engine(
                    chat_mode="context", verbose=True)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro", google_api_key="Your_Gemini_API_Key", temperature=0.1)
                st.session_state.langchain_chat_engine = ConversationChain(llm=llm, verbose=True,
                                                                           memory=ConversationBufferMemory())
            else:
                st.error(
                    "Failed to initialize chat engines due to data loading error.")
    except Exception as e:
        st.error(f"An error occurred while initializing chat engines: {e}")


# Function to handle user input and generate responses
def handle_user_input(prompt):
    try:
        response = st.session_state.chat_engine.chat(prompt).response
        template = ChatPromptTemplate.from_messages([
            ("system",
             "Responding as the AI Bot to provide accurate answers to user questions based on the provided context. Ensuring relevance of answers to the context provided. If the question seems irrelevant, please contact the developer. All answers provided are expected to be correct. Context: `{response}`"),
            ("human", st.session_state.messages[-1]["content"]),
            ("ai", st.session_state.messages[-2]["content"]),
            ("human", "{user_input}"),
        ])
        messages1 = template.format_messages(
            user_input=prompt, response=response)
        response = st.session_state.langchain_chat_engine.predict(input=messages1).replace("AIMessage(content='",
                                                                                           "").replace("')", "")
        return response
    except Exception as e:
        st.error(f"An error occurred while generating response: {e}")
        return None


# Main function
def main():
    try:
        # Initialize chat engines
        initialize_chat_engines()

        # Streamlit app title
        st.title("Chat With Document")

        # Initialize messages if not already in session state
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me questions ..."}]

        # User input prompt
        prompt = st.chat_input("Your question")

        # Add user input to session messages
        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Respond to user input
        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                response = handle_user_input(prompt)
                if response:
                    st.write(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Run the main function
if __name__ == "__main__":
    main()
