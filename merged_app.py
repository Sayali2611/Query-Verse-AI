from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import re
import json
import requests
import uuid
from werkzeug.utils import secure_filename
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import speech_recognition as sr
from io import BytesIO

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sigce.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Firebase
cred = credentials.Certificate("firebase-credentials.json")
firebase_admin.initialize_app(cred)
firestore_db = firestore.client()

# API Configuration
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_AI_URL = "https://api.together.xyz/v1/chat/completions"

# Load NLP Model
model = SentenceTransformer('all-mpnet-base-v2')

# Tesseract OCR Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_message = db.Column(db.String(500))
    bot_response = db.Column(db.String(500))
    intent = db.Column(db.String(50))
    session_id = db.Column(db.String(50))  # New field for session tracking

class LoginLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(50))
    success = db.Column(db.Boolean)

def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='admin@sigce.edu').first():
            admin = User(
                name='Admin',
                email='admin@sigce.edu',
                phone='+910000000000',
                password=generate_password_hash('Admin@123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()

# Initialize database
init_db()

# Helper function for phone validation
def validate_phone(phone):
    """Validate phone number format (international or Indian)"""
    pattern = r'^(\+91[\-\s]?)?[0]?(91)?[789]\d{9}$'
    return re.match(pattern, phone)

### -------- MAIN WEBSITE ROUTES -------- ###

@app.route('/')
def home():
    return render_template('indextemplate.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('index.html')

### -------- AUTHENTICATION API -------- ###

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    
    # Allow login with either email or phone
    login_id = data.get('login_id')  # Can be email or phone
    password = data.get('password')
    
    # Determine if login_id is email or phone
    if '@' in login_id:
        user = User.query.filter_by(email=login_id).first()
    else:
        user = User.query.filter_by(phone=login_id).first()
    
    login_log = LoginLog(
        user_id=user.id if user else None,
        ip_address=request.remote_addr,
        success=bool(user and check_password_hash(user.password, password))
    )
    db.session.add(login_log)
    
    if not user or not check_password_hash(user.password, password):
        db.session.commit()
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    user.last_login = datetime.utcnow()
    session.update({
        'user_id': user.id,
        'user_email': user.email,
        'user_name': user.name,
        'user_phone': user.phone,
        'is_admin': user.is_admin,
        'current_session': str(uuid.uuid4())  # Initialize a new session on login
    })
    db.session.commit()
    return jsonify({
        'success': True,
        'user': {
            'name': user.name,
            'email': user.email,
            'phone': user.phone,
            'is_admin': user.is_admin
        }
    })

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    
    # Validate required fields
    if not all([data.get('name'), data.get('email'), data.get('phone'), data.get('password')]):
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    # Validate email format
    if not re.match(r'[^@]+@[^@]+\.[^@]+', data.get('email')):
        return jsonify({'success': False, 'message': 'Invalid email format'}), 400
    
    # Validate phone number
    if not validate_phone(data.get('phone')):
        return jsonify({'success': False, 'message': 'Invalid phone number format. Use +91XXXXXXXXXX or 0XXXXXXXXXX'}), 400
    
    # Check if email or phone already exists
    if User.query.filter_by(email=data.get('email')).first():
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    if User.query.filter_by(phone=data.get('phone')).first():
        return jsonify({'success': False, 'message': 'Phone number already exists'}), 400
    
    # Create new user
    new_user = User(
        name=data.get('name'),
        email=data.get('email'),
        phone=data.get('phone'),
        password=generate_password_hash(data.get('password'))
    )
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Registration failed: ' + str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

### -------- ADMIN API -------- ###

@app.route('/api/admin/data')
def admin_data():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    users = User.query.all()
    login_logs = db.session.query(
        User.email,
        LoginLog.timestamp,
        LoginLog.success,
        LoginLog.ip_address
    ).join(LoginLog).order_by(LoginLog.timestamp.desc()).limit(50).all()
    
    return jsonify({
        'users': [{
            'id': u.id,
            'name': u.name,
            'email': u.email,
            'phone': u.phone,
            'registered_at': u.registered_at.strftime('%Y-%m-%d %H:%M'),
            'last_login': u.last_login.strftime('%Y-%m-%d %H:%M') if u.last_login else 'Never',
            'is_admin': u.is_admin
        } for u in users],
        'login_logs': [{
            'email': log.email,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M'),
            'success': log.success,
            'ip_address': log.ip_address
        } for log in login_logs]
    })

### -------- CHATBOT FUNCTIONALITY -------- ###

def save_chat_history(user_id, message, response, session_id=None):
    """Save chat messages to both Firebase and SQL database with session support."""
    user_doc_ref = firestore_db.collection('users').document(str(user_id))
    
    # Get or create user data in Firestore
    user_data = user_doc_ref.get()
    data = user_data.to_dict() if user_data.exists else {'chat_sessions': []}
    
    # Find or create the session
    session_id = session_id or session.get('current_session')
    current_session = None
    
    for s in data.get('chat_sessions', []):
        if s.get('session_id') == session_id:
            current_session = s
            break
    
    if not current_session:
        current_session = {
            'session_id': session_id,
            'created_at': datetime.utcnow().isoformat(),
            'history': []
        }
        if 'chat_sessions' not in data:
            data['chat_sessions'] = []
        data['chat_sessions'].append(current_session)
    
    # Add the new messages
    if message:
        current_session['history'].append({'user': message})
    if response:
        current_session['history'].append({'bot': response})
    
    # Save to Firestore
    user_doc_ref.set(data, merge=True)
    
    # Also save to SQL database
    if message and response:
        db.session.add(ChatHistory(
            user_id=user_id,
            user_message=message,
            bot_response=response,
            session_id=session_id
        ))
        db.session.commit()

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "data.json")
    if not os.path.exists(file_path):
        return {"queries": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"queries": []}

def save_data(data):
    file_path = os.path.join(os.path.dirname(__file__), "data.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON: {e}")

def format_response(text):
    text = re.sub(r'(\. )([A-Z])', r'.\n\2', text)
    return text.strip()

def get_ai_response(user_message, chat_history):
    """Get AI-generated response from Llama model."""
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [{"role": "user", "content": msg["user"]} for msg in chat_history if "user" in msg] + [{"role": "user", "content": user_message}],
        "max_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        response = requests.post(TOGETHER_AI_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        choices = response_data.get("choices", [])
        if choices and "message" in choices[0] and "content" in choices[0]["message"]:
            return format_response(choices[0]["message"]["content"])
        else:
            return "AI response not available."
    except requests.RequestException as e:
        print(f"AI request failed: {e}")
        return "Sorry, I couldn't generate a response at the moment."

def extract_text_from_image(image_path):
    """Extract text from image using OCR."""
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Image processing error: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"PDF processing error: {e}")
        return ""

def process_uploaded_file(file):
    """Process uploaded file and return extracted text."""
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    extracted_text = ""
    try:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            extracted_text = extract_text_from_image(file_path)
        elif filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()
    except Exception as e:
        print(f"File processing error: {e}")

    try:
        os.remove(file_path)  # Clean up the uploaded file
    except:
        pass

    return extracted_text.strip()

def get_best_match_semantic(user_question, data):
    """Find the best matching stored answer using NLP."""
    if not data or "queries" not in data or not data["queries"]:
        return None

    try:
        user_embedding = model.encode([user_question])
        question_embeddings = model.encode([q["question"] for q in data["queries"]])
        similarities = cosine_similarity(user_embedding, question_embeddings)[0]
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        return data["queries"][best_match_index] if best_match_score > 0.7 else None
    except Exception as e:
        print(f"Semantic matching error: {e}")
        return None

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get all chat sessions for the current user."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_doc_ref = firestore_db.collection('users').document(str(session['user_id']))
    user_data = user_doc_ref.get()
    
    if not user_data.exists:
        return jsonify({'sessions': []})
    
    data = user_data.to_dict()
    sessions = data.get('chat_sessions', [])
    
    # Sort sessions by creation date (newest first)
    sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return jsonify({
        'sessions': [{
            'id': s.get('session_id'),
            'created_at': s.get('created_at'),
            'preview': s['history'][0]['user'] if s.get('history') and len(s['history']) > 0 and 'user' in s['history'][0] else 'New Chat'
        } for s in sessions if s.get('session_id')]
    })

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for a specific session."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    user_doc_ref = firestore_db.collection('users').document(str(session['user_id']))
    user_data = user_doc_ref.get()
    
    if not user_data.exists:
        return jsonify({'history': []})
    
    data = user_data.to_dict()
    for s in data.get('chat_sessions', []):
        if s.get('session_id') == session_id:
            return jsonify({'history': s.get('history', [])})
    
    return jsonify({'history': []})

@app.route('/api/chat/new', methods=['POST'])
def new_chat_session():
    """Create a new chat session."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    new_session_id = str(uuid.uuid4())
    session['current_session'] = new_session_id
    
    # Initialize the session in Firestore
    user_doc_ref = firestore_db.collection('users').document(str(session['user_id']))
    user_data = user_doc_ref.get()
    data = user_data.to_dict() if user_data.exists else {'chat_sessions': []}
    
    if 'chat_sessions' not in data:
        data['chat_sessions'] = []
    
    data['chat_sessions'].append({
        'session_id': new_session_id,
        'created_at': datetime.utcnow().isoformat(),
        'history': []
    })
    
    user_doc_ref.set(data, merge=True)
    
    return jsonify({'success': True, 'session_id': new_session_id})

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    user_id = str(session['user_id'])
    user_message = data.get("message", "").strip()
    session_id = session.get('current_session')

    if not user_message:
        return jsonify({'response': 'No message received!'})

    if not session_id:
        return jsonify({'error': 'No active chat session'}), 400

    # Get current session history from Firestore
    user_doc_ref = firestore_db.collection('users').document(user_id)
    user_data = user_doc_ref.get()
    current_history = []
    
    if user_data.exists:
        data = user_data.to_dict()
        for s in data.get('chat_sessions', []):
            if s.get('session_id') == session_id:
                current_history = s.get('history', [])
                break

    # First check stored data.json (knowledge base)
    stored_data = load_data()
    matched_query = get_best_match_semantic(user_message, stored_data)

    if matched_query:
        response = format_response(matched_query["answer"])
    else:
        response = get_ai_response(user_message, current_history)

    # Save to both Firestore and SQL database
    save_chat_history(session['user_id'], user_message, response, session_id)

    # Store new Q&A in knowledge base if needed
    if not matched_query and response and user_message:
        if "queries" not in stored_data:
            stored_data["queries"] = []
        stored_data["queries"].append({"question": user_message, "answer": response})
        save_data(stored_data)

    return jsonify({'response': response})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload & process files with instructions."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"}), 400

    instructions = request.form.get("instructions", "Analyze this document and summarize the key points.")
    extracted_text = process_uploaded_file(file)

    if not extracted_text:
        return jsonify({"error": "No readable text found in file."}), 400

    # Create a prompt combining the instructions and extracted text
    prompt = f"{instructions}\n\nDocument content:\n{extracted_text}"
    
    # Get AI response based on the instructions
    ai_response = get_ai_response(prompt, [{"user": prompt}])

    # Save the file processing to chat history
    save_chat_history(
        session['user_id'],
        f"[File Upload] {file.filename}. Instructions: {instructions}",
        ai_response,
        session.get('current_session')
    )

    return jsonify({
        "extracted_text": extracted_text,
        "ai_response": ai_response
    })

@app.route('/api/voice', methods=['POST'])
def voice_to_text():
    """Convert voice recording to text."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    recognizer = sr.Recognizer()
    
    try:
        # Convert the FileStorage to a file-like object
        audio_data = BytesIO(audio_file.read())
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return jsonify({'text': text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {e}'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)