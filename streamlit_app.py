import streamlit as st
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon="💬", layout="wide")

# Initialize session state for messages if not already exists
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个编码助手"}
    ]

# Sidebar for API key and model selection
st.sidebar.title("🤖 Chat Configuration")
api_key = st.sidebar.text_input("Hugging Face API Key", type="password")
model_options = [
    "Qwen/Qwen2.5-Coder-32B-Instruct", 
    "Qwen/Qwen2.5-72B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-27b-it",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct"
]
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Temperature and max tokens sliders
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 50, 4096, 2048, 50)

def generate_response(messages, api_key, model, temperature, max_tokens):
    try:
        client = InferenceClient(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.7,
            stream=True
        )
        return stream  # Return the stream generator
    except Exception as e:
        return f"Error: {str(e)}"

# Main chat interface
st.title("🌐 Hugging Face Chat Assistant")

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your message"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if api_key:
            response_container = st.empty()  # Create an empty container for streaming
            full_response = ""  # Variable to hold the complete response

            with st.spinner("Generating response..."):
                # Generate response stream
                stream = generate_response(
                    st.session_state.messages, 
                    api_key, 
                    selected_model, 
                    temperature, 
                    max_tokens
                )

                if isinstance(stream, str) and stream.startswith("Error:"):
                    # Handle errors during API calls
                    st.error(stream)
                else:
                    try:
                        # Display chunks as they arrive
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                response_container.markdown(full_response)  # Update the container
                        
                        # Add final response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_response}
                        )
                    except Exception as e:
                        st.error(f"Error while streaming response: {str(e)}")
        else:
            st.warning("Please enter a Hugging Face API key in the sidebar.")

# Additional UI customization
st.sidebar.markdown("---")
st.sidebar.info("Configure your chat settings and enter your Hugging Face API key to start chatting!")
