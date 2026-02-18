QUERY VERSE AI
A College Enquiry AI Chatbot built with Flask, Firebase, and LLaMA 3 integration, designed to handle student queries, provide information, and streamline the enquiry process.

Features
AI Chatbot powered by LLaMA 3 for natural conversation
Flask Backend for smooth request handling
Firebase Integration for authentication & database storage
OCR Support for reading text from images
File Uploads to process and answer document-based queries
Chat History to keep track of past conversations
Dark Mode UI for a modern look
Project Structure
QueryVerse-AI/ │── app.py # Main Flask application │── requirements.txt # Python dependencies │── static/ # CSS, JS, and images │── templates/ # HTML templates │── firebase-credentials.json (ignored in repo)

Clone this repository

git clone https://github.com/aanchalmore2005/QueryVerse-AI.git
cd QueryVerse-AI
Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
Install dependencies

pip install -r requirements.txt
Run the application

python merged_app.py
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
