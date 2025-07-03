import os
import nest_asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import streamlit as st

# Apply nest_asyncio to fix event loop error
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get and validate Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY not found in .env file. Please set it and restart the application.")
    st.stop()

# Set

# Set up the provider and model
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai", 
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# Create the agent with optimized instructions
agent = Agent(
    name="Chat Agent",
    instructions="""
    You are a highly capable chat assistant designed to assist users with a wide variety of tasks and questions. 
    Your goal is to provide accurate, detailed, and helpful responses to any query the user might have. 
    You can assist with answering factual questions, explaining concepts, solving problems, writing and debugging code, generating creative content, and more. 
    You will be provided with the conversation history and the user's current message. 
    Use the history to maintain context and provide relevant responses. 
    If you need to ask for clarification, do so politely. 
    If the user says 'forget' or 'reset', acknowledge that you will clear the conversation history and start fresh, but do not include any previous messages in your response after clearing. 
    Always strive to be as helpful as possible, maintaining a friendly and engaging tone.
    """,
    model=model
)

# Streamlit app
st.title("Chat with Zeus Agent")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your chat assistant. Ask me anything, and I'll do my best to help! Type 'forget' or 'reset' to clear our conversation history."}
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check for forget/reset command
    if prompt.lower() in ["forget", "reset"]:
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversation history cleared. How can I assist you now?"}
        ]
        with st.chat_message("assistant"):
            st.markdown("Conversation history cleared. How can I assist you now?")
    else:
        # Prepare input for the agent
        input_messages = [{"role": "system", "content": agent.instructions}] + st.session_state.messages
        
        # Run the agent with the conversation history
        with st.chat_message("assistant"):
            try:
                result = Runner.run_sync(agent, input_messages)
                assistant_response = result.final_output
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")