import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
llm = OllamaLLM(model="mistral")          
# Session‑state chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Text‑to‑speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)            # Speaking speed (words per minute)

# Speech recogniser
recognizer = sr.Recognizer()

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def speak(text: str) -> None:
    """Read text aloud."""
    engine.say(text)
    engine.runAndWait()

def listen() -> str:
    """Capture a voice query and return it as lowercase text."""
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"👂 You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("🤖 Sorry, I couldn't understand. Try again!")
        return ""
    except sr.RequestError:
        st.write("⚠️ Speech‑recognition service unavailable")
        return ""

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation:\n{chat_history}\nUser: {question}\nAI:"
)

def run_chain(question: str) -> str:
    """Generate an AI reply and update chat memory."""
    chat_history_text = "\n".join(
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in st.session_state.chat_history.messages
    )

    response = llm.invoke(
        prompt.format(chat_history=chat_history_text, question=question)
    )

    # Store messages
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(str(response))

    return str(response)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.title("🤖 AI Voice Assistant ")
st.write("🎙️ Click the button below to speak to your AI assistant!")

if st.button("🎤 Start Listening"):
    user_query = listen()
    if user_query:
        ai_response = run_chain(user_query)
        st.write(f"**You:** {user_query}")
        st.write(f"**AI:** {ai_response}")
        speak(ai_response)                 # Speak the reply

st.subheader("📜 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")
