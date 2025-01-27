# MURBURN Chatbot

This project implements the **MURBURN Advisor Chatbot** using Flask and multiple AI technologies to facilitate biological and concept-based queries. The chatbot leverages **Google's Gemini 1.5 Pro** language model, **FAISS** for document similarity search, **SentenceTransformers** for embedding user queries, and **MongoDB** for storing related documents. It is designed to help users by providing classical biological concept explanations, along with a specialized **MURBURN** concept tailored to their query.

### Project Structure

1. **Flask Web Application**:
   - The core of the chatbot is built using the **Flask framework**, providing routes and handling requests.
   - The `/chat` route allows users to send a POST request with their query, which is processed by the chatbot and returns relevant information.

2. **MongoDB Setup**:
   - MongoDB is used to store PDF collections with biological data. This provides an efficient way to manage large datasets related to biological concepts and research.

3. **Groq Integration**:
   - Groq API is used for accelerating AI inference tasks. It assists in processing large datasets, providing faster response times for the chatbot.

4. **Google Generative AI (Gemini 1.5)**:
   - The chatbot generates responses using **Google's Gemini 1.5 Pro model**, a powerful generative AI, configured with settings optimized for biological concept discussions.

5. **FAISS Indexing**:
   - The **FAISS (Facebook AI Similarity Search)** library is used to create a similarity search index for efficient retrieval of related documents based on user queries. It uses **SentenceTransformer** embeddings to convert documents and queries into vector representations.

6. **Wikipedia Scraping**:
   - The chatbot integrates Wikipedia as an additional knowledge source. It uses **BeautifulSoup** for web scraping to retrieve related articles and present them as part of the response.

7. **MURBURN Concept**:
   - The chatbot's primary goal is to compare classical biological concepts with the **MURBURN** concept. The MURBURN concept is introduced as a specialized approach to explaining biological phenomena and is structured into four main sections:
     - **Classical Biology Concept**: General explanation of classical biological theory.
     - **MURBURN Concept**: The chatbot's own definition and application of the MURBURN concept.
     - **Key Differences**: Comparison between classical biology and MURBURN.
     - **Analogy**: Easy-to-understand analogy to clarify the differences.

### How It Works

1. **User Input**:
   - A user submits a query related to biology (e.g., “What is Photosynthesis?”) via a POST request to the `/chat` route.
   
2. **Response Generation**:
   - The system generates a **main response** based on the **MURBURN concept** using the **Google Gemini model**.
   - Simultaneously, it scrapes relevant results from **Wikipedia** for additional information.
   - The system combines these results and generates a **classification response** comparing the MURBURN concept with classical biology.

3. **Similarity Search**:
   - The query is passed to the **FAISS index**, which performs a similarity search using pre-embedded documents. The results from the index are appended to the response.

4. **Final Response**:
   - The chatbot returns a detailed response to the user, combining:
     - The **main response** from the AI model.
     - **Related articles** from Wikipedia and **similar documents** from the FAISS index.
     - A **classification** comparing the MURBURN concept and classical biology.

### Dependencies

The following Python libraries are used in this project:

- `Flask`: Web framework for building the API.
- `pymongo`: MongoDB client for data storage.
- `google-generativeai`: Google’s Gemini API for generative AI tasks.
- `langchain`: A framework for building LLM-powered applications.
- `FAISS`: Efficient similarity search library for large datasets.
- `SentenceTransformers`: Converts text into vector embeddings for FAISS and search tasks.
- `BeautifulSoup`: Web scraping library for retrieving data from Wikipedia.
- `cachetools`: Caching library for storing Wikipedia search results.
- `dotenv`: Load environment variables securely.

### Setup Instructions

To set up this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/smadhiva/MURBURN-CHATBOT.git
   cd murburn-advisor-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory of the project and add the following keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     GOOGLE_API_KEY=your_google_api_key
     ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. The Flask app will be running at `http://localhost:5000`.

### Usage

1. **Start the server**:
   - Run the Flask application locally to handle user queries.
   
2. **Send a POST request** to `/chat` with a JSON body:
   ```json
   {
     "user_prompt": "What is Photosynthesis?"
   }
   ```

3. **Receive the response**:
   - The chatbot will return a structured response with the classical biology explanation, the MURBURN concept comparison, and related articles or documents.

### Example Response

```json
{
  "main_response": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose molecules. It is essential for plant growth and the foundation of the food chain.",
  "related_cases": [
    {"title": "Photosynthesis - Wikipedia", "link": "https://en.wikipedia.org/wiki/Photosynthesis"}
  ],
  "classification": "The classical biology concept of Photosynthesis focuses on energy conversion in plants, while the MURBURN concept introduces a novel framework for understanding metabolic processes at the cellular level."
}
```

### License

This project is licensed under the MIT License.

