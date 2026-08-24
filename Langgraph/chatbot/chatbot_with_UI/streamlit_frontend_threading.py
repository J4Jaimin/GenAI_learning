import streamlit as st
import uuid
from langgraph_backend import chatbot, checkpointer
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

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].insert(0, thread_id)

def retrieve_all_threads():
    # every checkpoint tuple in the sqlite db carries its thread_id in config
    all_threads = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in all_threads:
            all_threads.append(thread_id)
    return all_threads

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])

    message_history = []
    for message in messages:
        role = 'user' if message.type == 'human' else 'assistant'
        message_history.append({'role': role, 'content': message.content})

    return message_history

def thread_label(thread_id):
    messages = load_conversation(thread_id)
    for message in messages:
        if message['role'] == 'user':
            return message['content'][:30]
    return thread_id[:8]

# ---------------------- session state init ----------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    # loaded from chatbot.db so past chats survive an app restart
    st.session_state['chat_threads'] = retrieve_all_threads()

# ---------------------- sidebar UI ----------------------

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(thread_label(thread_id), key=thread_id):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)

# ---------------------- main UI ----------------------

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # register this thread in the sidebar the first time it gets a message
    add_thread(st.session_state['thread_id'])

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

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
