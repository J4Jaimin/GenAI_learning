import streamlit as st
import uuid
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# ---------------------- chat bubble alignment (user: right, assistant: left) ----------------------

st.markdown("""
    <style>
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- utility functions ----------------------

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread(thread_id, title):
    if thread_id not in [t['thread_id'] for t in st.session_state['chat_threads']]:
        st.session_state['chat_threads'].insert(0, {'thread_id': thread_id, 'title': title})

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])

    message_history = []
    for message in messages:
        role = 'user' if message.type == 'human' else 'assistant'
        message_history.append({'role': role, 'content': message.content})

    return message_history

# ---------------------- session state init ----------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# ---------------------- sidebar UI ----------------------

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread in st.session_state['chat_threads']:
    if st.sidebar.button(thread['title'], key=thread['thread_id']):
        st.session_state['thread_id'] = thread['thread_id']
        st.session_state['message_history'] = load_conversation(thread['thread_id'])

# ---------------------- main UI ----------------------

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # register this thread in the sidebar the first time it gets a message
    add_thread(st.session_state['thread_id'], user_input[:30])

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=CONFIG,
            stream_mode="messages"
        ):

            if message_chunk.content:

                full_response += message_chunk.content

                message_placeholder.markdown(
                    full_response
                )

    ai_message = full_response
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
