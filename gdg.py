import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
X_train = ["Hello, how are you?", "What's your name?", "Goodbye!", "Tell me a joke."]
y_train = ["greeting", "greeting", "goodbye", "joke"]

# Train a TfidfVectorizer and Naive Bayes classifier
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

# Fit the model
X_train_vec = vectorizer.fit_transform(X_train)
classifier.fit(X_train_vec, y_train)

# Save the vectorizer and classifier
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(classifier, 'clf_model.pkl')

import random
import joblib
import streamlit as st

# Load the pre-trained vectorizer and classifier
vectorizer = joblib.load('vectorizer.pkl')  # Load vectorizer model
clf = joblib.load('clf_model.pkl')  # Load classifier model

# Define your intents
intents = [
    {
        "tag": "greeting",
        "patterns": [
            "Hello",
            "Hi",
            "Hey",
            "Good morning",
            "Good afternoon",
            "Good evening",
            "How are you?",
            "What's up?",
            "Howdy"
        ],
        "responses": [
            "Hello!",
            "Hi, how can I assist you?",
            "Hey there!",
            "Good to see you!",
            "Hi, how can I help today?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "Goodbye",
            "Bye",
            "See you",
            "Take care",
            "Catch you later",
            "Have a nice day"
        ],
        "responses": [
            "Goodbye!",
            "Take care!",
            "See you later!",
            "Have a great day!",
            "Catch you later!"
        ]
    },
    {
        "tag": "thanks",
        "patterns": [
            "Thank you",
            "Thanks",
            "Thanks a lot",
            "Thank you so much",
            "I appreciate it"
        ],
        "responses": [
            "You're welcome!",
            "No problem!",
            "Glad I could help!",
            "Anytime!",
            "You're welcome, happy to help!"
        ]
    },
    {
        "tag": "joke",
        "patterns": [
            "Tell me a joke",
            "Do you know any jokes?",
            "Make me laugh",
            "Tell me something funny",
            "Give me a joke"
        ],
        "responses": [
            "Why don't skeletons fight each other? They don't have the guts!",
            "What do you call fake spaghetti? An impasta!",
            "Why don’t oysters donate to charity? Because they are shellfish.",
            "I told my computer I needed a break, and it froze!",
            "Why did the coffee file a police report? It got mugged!"
        ]
    },
    {
        "tag": "age",
        "patterns": [
            "How old are you?",
            "What is your age?",
            "How many years old are you?",
            "How old is the chatbot?",
            "What’s your age?"
        ],
        "responses": [
            "I don't have an age, I'm just a chatbot!",
            "I am timeless!",
            "I don't age like humans do!",
            "Age is just a number, right?",
            "I don't have a birthday, but I was born from code!"
        ]
    },
    {
        "tag": "location",
        "patterns": [
            "Where are you located?",
            "Where do you live?",
            "Where are you from?",
            "What is your location?"
        ],
        "responses": [
            "I'm a chatbot, I live in the cloud!",
            "I reside in the digital world!",
            "I am everywhere and nowhere!",
            "I don't have a physical location, just a server!"
        ]
    },
    {
        "tag": "bot_info",
        "patterns": [
            "What is your name?",
            "Who are you?",
            "Tell me about yourself",
            "What do you do?"
        ],
        "responses": [
            "I am your friendly chatbot, here to help!",
            "I'm an AI chatbot created to assist you!",
            "I'm here to help you with your questions.",
            "I am a chatbot, created for fun and assistance!"
        ]
    },
    {
        "tag": "weather",
        "patterns": [
            "What's the weather like?",
            "Tell me the weather",
            "Is it going to rain today?",
            "What's the temperature outside?",
            "How's the weather?"
        ],
        "responses": [
            "I can't check real-time weather, but you can use weather websites or apps.",
            "Sorry, I don't have access to weather data right now.",
            "I'm not equipped with weather updates, but you can check your local forecast!"
        ]
    },
    {
        "tag": "news",
        "patterns": [
            "What's the news?",
            "Tell me the latest news",
            "What's happening in the world?",
            "Give me the news",
            "Any new updates?"
        ],
        "responses": [
            "I can't provide news updates at the moment, but you can check news websites for the latest info.",
            "Sorry, I am not connected to a news service right now.",
            "For the latest news, try visiting your favorite news outlet!"
        ]
    },
    {
        "tag": "help",
        "patterns": [
            "Can you help me?",
            "I need help",
            "Help me",
            "Can you assist me?",
            "I need assistance"
        ],
        "responses": [
            "Sure! How can I assist you?",
            "Of course! What do you need help with?",
            "I'm here to help! What do you need?",
            "How can I help you today?"
        ]
    },
    {
        "tag": "music",
        "patterns": [
            "What music do you like?",
            "Do you like music?",
            "Can you suggest some songs?",
            "What are your favorite tunes?",
            "Do you know any good songs?"
        ],
        "responses": [
            "I don't have personal preferences, but I can suggest some songs based on your taste!",
            "I love music! What genre do you prefer?",
            "I can't listen to music, but I can suggest some genres. What do you like?",
            "Music is a great way to relax! What are you in the mood for?"
        ]
    },
    {
        "tag": "quote",
        "patterns": [
            "Tell me a quote",
            "Give me an inspiring quote",
            "Can you give me some wisdom?",
            "Share a quote with me"
        ],
        "responses": [
            "“The only way to do great work is to love what you do.” – Steve Jobs",
            "“The future belongs to those who believe in the beauty of their dreams.” – Eleanor Roosevelt",
            "“In three words I can sum up everything I've learned about life: It goes on.” – Robert Frost",
            "“The best way to predict the future is to create it.” – Abraham Lincoln"
        ]
    }
]


# Define the chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])  # Convert input to vector
    tag = clf.predict(input_text)[0]  # Predict tag
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])  # Random response
            return response

# Streamlit main function
def main():
    st.title("Chatbot")
    st.write("Welcome to the chatbot! Type a message and press Enter.")

    # Maintain session state across inputs
    if 'counter' not in st.session_state:
        st.session_state.counter = 0

    st.session_state.counter += 1
    user_input = st.text_input("You:", key=f"user_input_{st.session_state.counter}")

    if user_input:
        # Get response from chatbot
        response = chatbot(user_input)

        # Display the chatbot's response
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{st.session_state.counter}")

        # Handle exit condition for the conversation
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
