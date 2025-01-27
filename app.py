import streamlit as st
import requests
import json

# Flask backend URL
FLASK_BACKEND_URL = "http://localhost:5000/chat"  # Adjust based on your Flask setup

# Streamlit UI components
st.title("MURBURN-BOT")

st.markdown("""
    Welcome to MURBURN-BOT, an advanced biology and concept analysis chatbot.
    """)

# Input text for user query
user_prompt = st.text_area("Ask your question:")

# Button to submit the query
if st.button("Ask"):
    if user_prompt:
        # Show progress bar
        with st.spinner("Processing your request..."):
            # Prepare the payload
            payload = {"user_prompt": user_prompt}

            # Send request to Flask API
            response = requests.post(FLASK_BACKEND_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()

                # Debug the response
                st.write("Debugging API Response:", data)

                # Display the results
                st.subheader("Main Response:")
                st.write(data.get("main_response", "No response provided."))

                st.subheader("Related Articles:")
                related_cases = data.get("related_cases", [])
                if related_cases:
                    for case in related_cases:
                        title = case.get("title", "No Title Provided")
                        link = case.get("link", "#")
                        st.write(f"- [{title}]({link})")
                else:
                    st.write("No related articles found.")

                st.subheader("Classification:")
                st.write(data.get("classification", "No classification provided."))
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
    else:
        st.warning("Please enter a question.")
