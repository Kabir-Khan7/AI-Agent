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
    Now I am providing you the data about Kabir Khurshid below is the data of Kabir whenever user asks question about Kabir answer from this data:
    Kabir Khurshid: Personal and Professional Profile

Kabir Khurshid was born on May 11, 2003, in Karachi, Pakistan, where he has lived his entire life. He is a driven young professional focusing on supply chain management and finance, with a strong interest in technology and business. This profile summarizes his background from early education through to his current career, highlighting his experiences, skills, and goals.
Education

Kabir attended local schools in Karachi for his early education, demonstrating consistent academic dedication. His schooling included:

    Diamond City School (Karachi) — Nursery to Class 4.

    The Educator School (Karachi) — Classes 5 to 7.

    Karachi Montessori Grammar School — Class 8.

    The Smart School (Karachi) — Classes 9 and 10 (Matriculation completed in 2020).

After matriculation, Kabir pursued higher secondary education at Government Degree College Gulzar-e-Hijri in Karachi, majoring in F.Sc. (Pre-Engineering) and completing it in 2023. He then took a one-year gap before enrolling in university. In Fall 2024, Kabir was admitted to an Associate Degree Program in Accounting and Finance at Hamdard University (Pakistan), which he is currently studying.
Professional Experience

Kabir has gained practical experience through internships and professional roles in supply chain and procurement. His work history includes:

    Daraz (Karachi) — Supply Chain Intern (Mar 2022 – Apr 2022).

    The Hub Leather (Karachi) — Procurement Intern (Aug 2022 – Nov 2022).

    Supertouch (Karachi) — Inventory Analyst (Jan 2023 – Nov 2023).

    Level 3 BOS (Karachi) — Supply Chain Executive (Jan 2024 – present; promoted to Senior Executive).

In his current role at Level 3 BOS, Kabir oversees various supply chain operations, including inventory management and logistics coordination. He often works night shifts and takes on significant responsibilities to ensure smooth business operations. Balancing these work duties with his ongoing studies demonstrates his strong dedication and resilience. These professional experiences have helped him develop robust problem-solving skills and a solid understanding of corporate processes.
Skills and Interests

Kabir combines fitness, technology, and analytical skills in his personal and professional life. His key interests and abilities include:

    Fitness and Well-being: A gym enthusiast, Kabir regularly works out to stay healthy and energized.

    Technology and Finance: Passionate about tech innovations and financial strategies, always exploring how they intersect with business needs.

    Business and Supply Chain: Skilled in business management, supply chain logistics, and accounting principles from both his education and hands-on roles.

    Programming and Data Science: Competent in programming fundamentals and currently learning agentic AI and data science to apply technological solutions in business contexts.

    Continued Learning: Completed a Corporate Management course at the MTF Institute of Technology & Finance (Lisbon) online, enhancing his managerial and strategic knowledge.

Languages

    English: Fluent (native proficiency).

    Spanish: Currently learning (beginner to intermediate level).

    Chinese: Currently learning (beginner level).

Goals and Aspirations

Kabir’s long-term goal is to leverage his skills and experiences to make a significant impact in the business world. He aspires to:

    Establish his own tech-driven business that provides innovative solutions to solve other companies’ problems.

    Combine his corporate management training and passion for finance to create services that improve organizational efficiency and performance.

Kabir is known for his discipline and perseverance. Balancing a demanding job (often including night shifts) with ongoing education has taught him the value of hard work and adaptability. He remains focused on continuous growth, aiming to use his knowledge and skills to help businesses succeed and to achieve his entrepreneurial ambitions.
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