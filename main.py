from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import google.generativeai as genai
import random
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the key
apikey = os.getenv("API_KEY")
# Flask setup
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Configure Gemini API
genai.configure(api_key=apikey)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Q-table
Q_table = {}

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate (for random responses)


def get_best_response(query):
    """Choose the best response based on the Q-table."""
    if query in Q_table and random.random() > epsilon:  # Exploit known good response
        return max(Q_table[query], key=Q_table[query].get)
    else:  # Explore new response
        return None


def update_q_table(query, response, reward):
    """Update the Q-table using the Q-learning formula."""
    if query not in Q_table:
        Q_table[query] = {}

    if response not in Q_table[query]:
        Q_table[query][response] = 0  # Initialize response Q-value

    # Q-learning formula: Q(s, a) = Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]
    old_value = Q_table[query][response]
    max_future_q = max(Q_table[query].values(), default=0)  # Best future reward
    new_q_value = old_value + alpha * (reward + gamma * max_future_q - old_value)

    Q_table[query][response] = new_q_value  # Update Q-value


@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint to interact with the chatbot."""
    data = request.json
    query = data.get('query', '').strip()
    reward = data.get('reward', 0)  # Getting the feedback (reward) from the frontend

    if query.lower() == "exit":
        return jsonify({'response': 'Goodbye!'})

    # Check if a known response exists
    best_response = get_best_response(query)

    if best_response:
        response = best_response
    else:
        # Generate a new response using Gemini API
        full_query = query + " Give this as a short manner of its possible disease symptoms and remedies in para tag format."
        response = model.generate_content(full_query).text

    # Update the Q-table with the reward received (feedback)
    update_q_table(query, response, reward)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
