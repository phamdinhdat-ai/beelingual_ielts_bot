# ui.py
import streamlit as st
import requests
import json # For pretty printing JSON responses

# --- Configuration ---
FASTAPI_BASE_URL = "http://localhost:8333" # Your FastAPI app's URL

# --- Helper Functions to Interact with API ---

def register_user(username, email, password):
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/auth/register",
            json={"username": username, "email": email, "password": password}
        )
        return response
    except requests.exceptions.ConnectionError:
        return None

def login_user(username, password):
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/auth/token",
            data={"username": username, "password": password} # Form data for token endpoint
        )
        return response
    except requests.exceptions.ConnectionError:
        return None

def get_user_me(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{FASTAPI_BASE_URL}/users/me", headers=headers)
        return response
    except requests.exceptions.ConnectionError:
        return None

def upload_document(file_uploader, user_type="user", token=None, guest_session_id=None):
    if file_uploader is not None:
        files = {"file": (file_uploader.name, file_uploader.getvalue(), file_uploader.type)}
        headers = {}
        url = ""

        if user_type == "user" and token:
            headers["Authorization"] = f"Bearer {token}"
            url = f"{FASTAPI_BASE_URL}/documents/user"
        elif user_type == "guest" and guest_session_id:
            url = f"{FASTAPI_BASE_URL}/documents/guest/{guest_session_id}"
        else:
            st.error("Invalid parameters for document upload.")
            return None

        try:
            response = requests.post(url, files=files, headers=headers)
            return response
        except requests.exceptions.ConnectionError:
            return None
    return None


def send_chat_message(query, user_type="user", token=None, session_id=None, guest_session_id=None):
    payload = {"query": query}
    headers = {}
    url = ""

    if user_type == "user" and token:
        headers["Authorization"] = f"Bearer {token}"
        payload["session_id"] = session_id # Can be None, API generates if so
        url = f"{FASTAPI_BASE_URL}/chat/user"
    elif user_type == "guest":
        payload["guest_session_id"] = guest_session_id # Can be None for first message
        url = f"{FASTAPI_BASE_URL}/chat/guest"
    else:
        st.error("Invalid parameters for chat message.")
        return None

    try:
        response = requests.post(url, json=payload, headers=headers)
        return response
    except requests.exceptions.ConnectionError:
        return None

def end_guest_session_api(guest_session_id):
    if not guest_session_id:
        return None
    try:
        response = requests.delete(f"{FASTAPI_BASE_URL}/chat/guest/{guest_session_id}")
        return response
    except requests.exceptions.ConnectionError:
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="Agentic API UI", layout="wide")
st.title("🤖 Agentic System Interaction UI")

# --- Session State Management ---
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'guest_session_id' not in st.session_state:
    st.session_state.guest_session_id = None
if 'chat_history_user' not in st.session_state:
    st.session_state.chat_history_user = [] # List of (query, response, suggestions)
if 'chat_history_guest' not in st.session_state:
    st.session_state.chat_history_guest = [] # List of (query, response, suggestions)
if 'user_chat_session_id' not in st.session_state: # For stateful user chat sessions
    st.session_state.user_chat_session_id = None


# --- Sidebar for Authentication and Mode Selection ---
st.sidebar.header("Mode & Authentication")
app_mode = st.sidebar.radio("Select Mode", ("User (Authenticated)", "Guest"))

if app_mode == "User (Authenticated)":
    st.sidebar.subheader("User Authentication")
    if st.session_state.auth_token:
        st.sidebar.success(f"Logged in as: {st.session_state.user_info.get('username', 'N/A')}")
        if st.sidebar.button("Logout"):
            st.session_state.auth_token = None
            st.session_state.user_info = None
            st.session_state.chat_history_user = []
            st.session_state.user_chat_session_id = None
            st.rerun() # Use st.rerun() in newer Streamlit versions
    else:
        login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])
        with login_tab:
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if not login_username or not login_password:
                    st.error("Username and password are required.")
                else:
                    res = login_user(login_username, login_password)
                    if res is None:
                        st.error("Connection Error: FastAPI server might not be running.")
                    elif res.status_code == 200:
                        token_data = res.json()
                        st.session_state.auth_token = token_data.get("access_token")
                        # Fetch user info
                        user_res = get_user_me(st.session_state.auth_token)
                        if user_res and user_res.status_code == 200:
                            st.session_state.user_info = user_res.json()
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        try:
                            st.error(f"Login failed: {res.json().get('detail', res.text)}")
                        except json.JSONDecodeError:
                            st.error(f"Login failed: {res.status_code} - {res.text}")


        with register_tab:
            reg_username = st.text_input("Username", key="reg_user")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register"):
                if not reg_username or not reg_email or not reg_password:
                    st.error("All fields are required for registration.")
                else:
                    res = register_user(reg_username, reg_email, reg_password)
                    if res is None:
                        st.error("Connection Error: FastAPI server might not be running.")
                    elif res.status_code == 201:
                        st.success("Registered successfully! Please login.")
                    else:
                        try:
                            st.error(f"Registration failed: {res.json().get('detail', res.text)}")
                        except json.JSONDecodeError:
                             st.error(f"Registration failed: {res.status_code} - {res.text}")


elif app_mode == "Guest":
    st.sidebar.subheader("Guest Session")
    if st.session_state.guest_session_id:
        st.sidebar.info(f"Guest Session ID: {st.session_state.guest_session_id}")
        if st.sidebar.button("End Guest Session & Clear Data"):
            res = end_guest_session_api(st.session_state.guest_session_id)
            if res is None:
                st.error("Connection Error: FastAPI server might not be running.")
            elif res.status_code == 204:
                st.success("Guest session ended and data cleaned up successfully.")
            else:
                st.error(f"Failed to end guest session: {res.status_code} - {res.text}")
            st.session_state.guest_session_id = None
            st.session_state.chat_history_guest = []
            st.rerun()
    else:
        st.sidebar.write("A guest session will start with your first message or document upload.")

# --- Main Content Area ---
st.header(f"{app_mode} Chat")

# Document Upload Section
st.subheader("Upload Document")
uploaded_file = st.file_uploader("Choose a file (e.g., .txt)", type=["txt", "md", "pdf"]) # Add more types if your backend supports them

if uploaded_file is not None:
    if st.button(f"Upload '{uploaded_file.name}'"):
        response = None
        if app_mode == "User (Authenticated)" and st.session_state.auth_token:
            response = upload_document(uploaded_file, user_type="user", token=st.session_state.auth_token)
        elif app_mode == "Guest":
            if not st.session_state.guest_session_id:
                # For guests, first document upload might also initiate the session via the chat endpoint logic
                # or you might need a dedicated "start guest session" API if docs are uploaded before first chat.
                # For simplicity, let's assume chat starts session, or we need to handle this.
                # A simple way: if guest_session_id is None, make a first chat call to establish it, then upload.
                # This is a bit clunky. Ideally, document upload to guest should also init session.
                # Let's assume your `/documents/guest/{guest_session_id}` requires a pre-existing session_id.
                # One way to handle this: if no guest_session_id, prompt user to send a message first
                # Or, if your backend creates the session on first doc upload for a new guest_id, it's fine.
                # Let's rely on the chat endpoint to create/manage guest_session_id.
                # If guest uploads doc first, they'll need to manually enter a new UUID or we force chat first.
                 st.warning("Please start a chat to get a Guest Session ID before uploading documents for a guest session.")

            if st.session_state.guest_session_id:
                response = upload_document(uploaded_file, user_type="guest", guest_session_id=st.session_state.guest_session_id)
            else:
                 st.warning("No active guest session. Send a message first or provide a Guest Session ID.")


        if response is None and (app_mode != "Guest" or st.session_state.guest_session_id is not None): # Avoid error if guest needs to start session
            st.error("Connection Error: FastAPI server might not be running.")
        elif response is not None:
            if response.status_code == 201:
                st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
                st.json(response.json())
            else:
                try:
                    st.error(f"Document upload failed: {response.json().get('detail', response.text)}")
                except json.JSONDecodeError:
                    st.error(f"Document upload failed: {response.status_code} - {response.text}")


# Chat Interface
st.subheader("Chat with Agent")

# Display Chat History
current_chat_history = []
if app_mode == "User (Authenticated)" and st.session_state.auth_token:
    current_chat_history = st.session_state.chat_history_user
elif app_mode == "Guest":
    current_chat_history = st.session_state.chat_history_guest

for i, (query, response_text, suggestions) in enumerate(current_chat_history):
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Agent:** {response_text}")
    if suggestions:
        st.markdown("**Suggested Questions:**")
        for sug_idx, sug_q in enumerate(suggestions):
            # Use a unique key for each button
            button_key = f"suggest_btn_{app_mode}_{i}_{sug_idx}"
            if st.button(sug_q, key=button_key):
                # When button is clicked, set its text as the new query
                st.session_state.chat_input_value = sug_q # Pre-fill input box
                st.rerun() # Rerun to process the new input (if input is tied to state)
                           # Or directly call chat function (simpler for this example)

    st.markdown("---")

# Chat Input
if 'chat_input_value' not in st.session_state:
    st.session_state.chat_input_value = ""

query_input = st.text_input("Your message:", value=st.session_state.chat_input_value, key="chat_input")

if st.button("Send Message"):
    if not query_input:
        st.warning("Please enter a message.")
    else:
        response_data = None
        if app_mode == "User (Authenticated)":
            if st.session_state.auth_token:
                res = send_chat_message(query_input, user_type="user", token=st.session_state.auth_token, session_id=st.session_state.user_chat_session_id)
                if res is None: st.error("Connection Error.")
                elif res.status_code == 200:
                    response_data = res.json()
                    st.session_state.user_chat_session_id = response_data.get("session_id") # Update session ID
                    st.session_state.chat_history_user.append((query_input, response_data.get("response"), response_data.get("suggested_questions")))
                else:
                    try: st.error(f"Chat failed: {res.json().get('detail', res.text)}")
                    except: st.error(f"Chat failed: {res.status_code} - {res.text}")
            else:
                st.warning("Please login to chat as a user.")
        elif app_mode == "Guest":
            res = send_chat_message(query_input, user_type="guest", guest_session_id=st.session_state.guest_session_id)
            if res is None: st.error("Connection Error.")
            elif res.status_code == 200:
                response_data = res.json()
                st.session_state.guest_session_id = response_data.get("guest_session_id") # Update/confirm session ID
                st.session_state.chat_history_guest.append((query_input, response_data.get("response"), response_data.get("suggested_questions")))
            else:
                try: st.error(f"Chat failed: {res.json().get('detail', res.text)}")
                except: st.error(f"Chat failed: {res.status_code} - {res.text}")

        st.session_state.chat_input_value = "" # Clear input box
        st.rerun() # Refresh to show new message

# For debugging: Show raw API responses (optional)
# if response_data:
#     st.subheader("Last API Response (Debug)")
#     st.json(response_data)