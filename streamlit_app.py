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
    "meta-llama/Llama-3.3-70B-Instruct", 
    "meta-llama/Llama-3.3-70B-Instruct"
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
        
        # Collect streamed response
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        
        return full_response
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
        with st.spinner("Generating response..."):
            # Only generate response if API key is provided
            if api_key:
                response = generate_response(
                    st.session_state.messages, 
                    api_key, 
                    selected_model, 
                    temperature, 
                    max_tokens
                )
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            else:
                st.warning("Please enter a Hugging Face API key in the sidebar.")

# Additional UI customization
st.sidebar.markdown("---")
st.sidebar.info("Configure your chat settings and enter your Hugging Face API key to start chatting!")
