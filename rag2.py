from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import os
from groq import Groq
from pymongo import MongoClient
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from werkzeug.utils import secure_filename
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Initialize the app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_database"
COLLECTION_NAME = "pdf_collection"

# Ensure the MongoDB client is not overwritten
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)  # Renamed to avoid conflict with mongo_client

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.1,
    model_kwargs={
        "max_output_tokens": 8192,
        "top_k": 10,
        "top_p": 0.95
    }
)

PROMPT_TEMPLATE = '''You are an advanced biology and concept analysis chatbot. Your task is to explain classical biology concepts related to the user's query, followed by an explanation of the MURBURN concept. Additionally, you will classify the key differences between the MURBURN concept and the user's query and provide an analogy to help clarify the concept. Ensure the explanations are clear, concise, and educational.

Follow this structure for your response:

1. **Classical Biology Concept**: 
   - Explain the classical biological concept related to the user's query, including its definition and relevance.
   - Discuss any foundational principles, laws, or theories that are important for understanding the concept.

2. **MURBURN Concept** (specific to the user's query):
   - Introduce and explain the MURBURN concept in the context of the user's query. Explain how it applies or contrasts with the query.
   - Highlight its unique characteristics, and how it may differ from classical biology concepts.
   - Explain the application or understanding of the MURBURN concept within the specific biological or scientific context related to the user's query.

3. **Key Differences**:
   - Classify and elaborate on the key differences between the classical biology concept and the MURBURN concept.
   - Provide specific distinctions, including scientific, theoretical, or practical aspects, showing how they diverge in principles, applications, or interpretations.

4. **Analogy**:
   - Provide an easy-to-understand analogy that contrasts the classical biology concept with the MURBURN concept.
   - The analogy should be simple and relatable, helping users grasp the differences in an intuitive way.
'''

# Embed queries using SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models too

# FAISS class to interact with the index
class FAISS:
    def __init__(self, index, docstore, index_to_docstore_id):
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    def similarity_search(self, query, k=5):
        query_vector = self.embed_query(query)  # Embed query
        D, I = self.index.search(query_vector, k)  # D is distances, I is indices
        results = []
        for i in I[0]:
            doc_id = self.index_to_docstore_id.get(i, None)
            if doc_id is not None:
                results.append(self.docstore[doc_id])
        return results

    def embed_query(self, query):
        """
        Convert the user query into an embedding (vector representation).
        """
        query_vector = embedding_model.encode([query])  # Returns a 2D array (1, dim)
        return np.array(query_vector, dtype=np.float32)  # Ensure it's in float32

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate FAISS index and save it
def generate_faiss_index(data, save_path="faiss_index.index"):
    """
    Generate a FAISS index from the provided data.

    Parameters:
        data (list of str): Text data to embed and index.
        save_path (str): Path to save the FAISS index.

    Returns:
        None
    """
    embeddings = embedding_model.encode(data)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, save_path)

    # Save docstore and index_to_docstore_id
    docstore = {i: {"text": doc} for i, doc in enumerate(data)}
    index_to_docstore_id = {i: i for i in range(len(data))}

    with open("docstore.pickle", "wb") as f:
        pickle.dump(docstore, f)

    with open("index_to_docstore_id.pickle", "wb") as f:
        pickle.dump(index_to_docstore_id, f)

    print("FAISS index and associated files generated and saved.")

# Function to load vector store (FAISS index and associated docstore)
def load_vector_store():
    try:
        if os.path.exists("faiss_index.index") and os.path.exists("docstore.pickle") and os.path.exists("index_to_docstore_id.pickle"):
            # Load FAISS index
            index = faiss.read_index("faiss_index.index")
            print("FAISS index loaded successfully.")

            # Load the docstore and index_to_docstore_id
            with open("docstore.pickle", "rb") as f:
                docstore = pickle.load(f)

            with open("index_to_docstore_id.pickle", "rb") as f:
                index_to_docstore_id = pickle.load(f)

            return FAISS(index, docstore, index_to_docstore_id)
        else:
            print("FAISS index or associated files not found. Generating new index...")
            # Generate new index with placeholder data
            data = ["Example document 1", "Example document 2", "Example document 3"]
            generate_faiss_index(data)
            return load_vector_store()
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

# Route for the homepage
@app.route('/')
def home():
    return "Welcome to the MURBURN Advisor Chatbot!"

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    try:
        # Validate input data
        data = request.json
        if not data or 'user_prompt' not in data:
            return jsonify({"error": "User prompt is required."}), 400

        user_prompt = data['user_prompt'].strip()

        # Handle greetings
        greetings = {"hi", "hii", "hello", "hey", "hola", "namaste"}
        if user_prompt.lower() in greetings:
            return jsonify({
                "main_response": "Welcome to the MURBURN Advisor Chatbot! How can I assist you today?",
                "related_cases": [],
                "classification": "N/A"
            }), 200

        # Generate main response from the language model
        main_response = generate_response_with_prompt(user_prompt)

        # Scrape Wikipedia for relevant results
        wiki_results = scrape_wikipedia_with_cache(user_prompt)
        if "error" in wiki_results:
            return jsonify({"error": wiki_results["error"]}), 500
        
        wikipedia_text = "\n".join([res["title"] for res in wiki_results])

        # Generate classification response based on main and Wikipedia results
        classification_response = generate_response_with_prompt2(main_response, wikipedia_text)

        # Load vector store and perform similarity search
        vector_store = load_vector_store()
        vector_store_results = []
        if vector_store:
            vector_store_results = vector_store.similarity_search(user_prompt)

        # Combine Wikipedia and vector store results
        combined_results = wiki_results + vector_store_results

        # Final response preparation
        if combined_results:
            return jsonify({
                "main_response": f"{main_response}\n\nHere are some related articles and information for '{user_prompt}':",
                "related_cases": combined_results,
                "classification": classification_response
            }), 200

        # Fallback for no results
        return jsonify({
            "main_response": f"{main_response}\n\nUnfortunately, no related articles or information were found for '{user_prompt}'.",
            "related_cases": [],
            "classification": classification_response
        }), 200

    except Exception as e:
        print(f"Error in /chat route: {e}")
        return jsonify({"error": "Internal server error."}), 500



# Function to generate response with prompt
def generate_response_with_prompt(user_query):
    try:
        formatted_prompt = PROMPT_TEMPLATE.replace("{user_query}", user_query)
        response = llm.invoke(formatted_prompt)  # Use invoke instead of directly calling llm
        return response
    except Exception as e:
        print(f"Error generating response with prompt: {e}")
        return "Error generating response."

def generate_response_with_prompt2(main_response, wikitext):
    """
    Generate an explanation comparing the MURBURN response with classical biology concepts
    using the main_response and wikitext.

    Parameters:
        main_response (str): Explanation focused on the MURBURN concept.
        wikitext (str): Classical biology-related information from Wikipedia.

    Returns:
        str: Structured comparison between MURBURN and classical biology.
    """
    try:
        # Define the comparison prompt template
        prompt_template = PromptTemplate(
            input_variables=["murburn_response", "classical_biology"],
            template="""
                Compare the following two explanations in detail:

                1. **MURBURN Concept Explanation**:
                   {murburn_response}

                2. **Classical Biology Explanation**:
                   {classical_biology}

                Provide a structured comparison by:
                - Highlighting similarities, if any.
                - Explaining the differences in concepts, principles, and applications.
                - Discussing how MURBURN diverges from classical approaches in biology.
                - Suggesting contexts where each explanation is most applicable.

                Finally, summarize the key takeaways in a concise paragraph.
            """
        )

        # Format the prompt
        formatted_prompt = prompt_template.format(
            murburn_response=main_response, classical_biology=wikitext
        )

        # Use the language model to generate a response
        response = llm.invoke(formatted_prompt)

        return response

    except Exception as e:
        print(f"Error in generate_response_with_prompt2: {e}")
        return "Error generating response."


# Function to scrape Wikipedia with caching
def scrape_wikipedia_with_cache(query):
    from cachetools import TTLCache
    
    cache = TTLCache(maxsize=100, ttl=3600)
    if query in cache:
        return cache[query]

    base_url = "https://en.wikipedia.org/w/index.php"
    params = {"search": query}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        search_results = soup.select(".mw-search-result-heading")
        for result in search_results[:5]:
            title = result.get_text(strip=True)
            link = "https://en.wikipedia.org" + result.find("a")["href"]
            results.append({"title": title, "link": link})

        cache[query] = results
        return results
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
