###basic ai agent with web ui
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI model
llm = OllamaLLM(model="mistral")

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Define AI chat prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation:\n{chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    # Retrieve past chat history
    chat_history_text = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages]
    )

    # AI response generation
    response = llm.invoke(
        prompt.format(chat_history=chat_history_text, question=question)
    )

    # Store user and AI messages in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return str(response)  # ✅ Correctly inside the function

# Streamlit UI
st.title("🤖 AI Chatbot with Memory")    
st.write("Ask me anything!")

user_input = st.text_input("Your question:")
if user_input:
    response = run_chain(user_input)
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {response}")

# Show chat history
st.subheader("📜 Chat History")
for msg in st.session_state.chat_history.messages:
    role = "🧑 You" if msg.type == "human" else "🤖 AI"
    st.markdown(f"**{role}:** {msg.content}")
