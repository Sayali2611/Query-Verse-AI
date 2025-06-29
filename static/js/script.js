document.addEventListener("DOMContentLoaded", function () {
    // DOM Elements
    const userInput = document.getElementById("user-input");
    const emojiPicker = document.getElementById("emoji-picker");
    const emojiContainer = document.querySelector('.emoji-list');
    const menuBtn = document.getElementById('menuBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const overlay = document.getElementById('overlay');
    const newChatBtn = document.getElementById('newChatBtn');
    const historyItems = document.getElementById('historyItems');
    const loadingIndicator = document.getElementById('loading-indicator');
    const fileUpload = document.getElementById('file-upload');
    const clearChatBtn = document.querySelector('.clear-chat-btn');
    const darkModeBtn = document.querySelector('.settings-btn');
    const sendBtn = document.querySelector('.send-btn');
    const speechBtn = document.querySelector('.speech-btn');
    

    // Chat state
    let currentSessionId = null;
    let isRecognizing = false;

    // Initialize the app
    function init() {
        // Check for saved dark mode preference
        if (localStorage.getItem('darkMode') === 'enabled') {
            document.body.classList.add('dark-mode');
        }
        
        checkAuthStatus();
        setupEventListeners();
        loadEmojis();
    }

    // Check if user is authenticated
    function checkAuthStatus() {
        fetch('/api/chat/sessions', {
            method: 'GET',
            credentials: 'same-origin'
        })
        .then(response => {
            if (response.status === 401) {
                window.location.href = '/';
            }
            return response.json();
        })
        .then(data => {
            if (data.sessions && data.sessions.length > 0) {
                loadChatSessions(data.sessions);
            } else {
                createNewSession();
            }
        })
        .catch(error => {
            console.error('Auth check failed:', error);
        });
    }

    // Set up all event listeners
    function setupEventListeners() {
        // Sidebar controls
        menuBtn.addEventListener('click', toggleSidebar);
        closeSidebarBtn.addEventListener('click', toggleSidebar);
        overlay.addEventListener('click', toggleSidebar);
        
        // Chat controls
        newChatBtn.addEventListener('click', createNewSession);
        clearChatBtn.addEventListener('click', clearChat);
        sendBtn.addEventListener('click', sendMessage);
        
        // Input handling
        userInput.addEventListener('keydown', handleInputKeys);
        userInput.addEventListener('input', autoResizeTextarea);
        
        // Emoji picker
        document.addEventListener('keydown', handleEscapeKey);
        document.getElementById('emoji-btn').addEventListener('click', toggleEmojiPicker);

        
        // File upload
        fileUpload.addEventListener('change', handleFileUpload);
        
        // Dark mode toggle
        darkModeBtn.addEventListener('click', toggleDarkMode);

        // Voice recognition
        speechBtn.addEventListener('click', toggleSpeechRecognition);


        document.addEventListener('click', (e) => {
            if (!emojiPicker.contains(e.target) && e.target.id !== 'emoji-btn') {
                emojiPicker.style.display = 'none';
            }
        });

        document.querySelectorAll('.quick-options button').forEach(button => {
            button.addEventListener('click', () => {
                const topic = button.textContent.trim();
                sendQuickMessage(topic);
            });
        
    })

    }

    // Load emojis into picker
    function loadEmojis() {
        const emojiList = ['😀', '😂', '😍', '😎', '😢', '😡', '😜', '🤔', '😇'];
        emojiList.forEach(emoji => {
            const emojiButton = document.createElement('button');
            emojiButton.textContent = emoji;
            emojiButton.onclick = () => addEmoji(emoji);
            emojiContainer.appendChild(emojiButton);
        });
    }

    // Toggle sidebar visibility
    function toggleSidebar() {
        historySidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    }

    // Create a new chat session
    function createNewSession() {
        fetch('/api/chat/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                currentSessionId = data.session_id;
                clearChat();
                loadChatHistory(currentSessionId);
                
                if (window.innerWidth < 768) {
                    toggleSidebar();
                }
                
                fetchChatSessions();
            }
        })
        .catch(error => {
            console.error('Error creating new session:', error);
            appendMessage('⚠ Error creating new chat session', 'bot');
        });
    }

    // Fetch all chat sessions
    function fetchChatSessions() {
        fetch('/api/chat/sessions')
        .then(response => response.json())
        .then(data => {
            if (data.sessions) {
                loadChatSessions(data.sessions);
            }
        })
        .catch(error => {
            console.error('Error fetching sessions:', error);
        });
    }

    // Load chat sessions into sidebar
    function loadChatSessions(sessions) {
        historyItems.innerHTML = '';
        
        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = `history-item ${session.id === currentSessionId ? 'active' : ''}`;
            item.textContent = session.preview;
            item.dataset.sessionId = session.id;
            
            item.addEventListener('click', () => {
                currentSessionId = session.id;
                loadChatHistory(currentSessionId);
                
                if (window.innerWidth < 768) {
                    toggleSidebar();
                }
            });
            
            historyItems.appendChild(item);
        });
    }

    // Load chat history for a session
    function loadChatHistory(sessionId) {
        fetch(`/api/chat/history?session_id=${sessionId}`)
        .then(response => response.json())
        .then(data => {
            clearChat();
            
            if (data.history && data.history.length > 0) {
                document.querySelector('.welcome-message').style.display = 'none';
                
                data.history.forEach(message => {
                    if (message.user) {
                        appendMessage(message.user, 'user');
                    }
                    if (message.bot) {
                        appendMessage(message.bot, 'bot');
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error loading chat history:', error);
        });
    }

    // Clear current chat UI
    function clearChat() {
        document.getElementById('chat-box').innerHTML = '';
        document.querySelector('.welcome-message').style.display = 'block';
    }

    // Handle keyboard input
    function handleInputKeys(e) {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                return; // Allow new line with Shift+Enter
            } else {
                e.preventDefault();
                sendMessage();
            }
        }
    }

    // Auto-resize textarea
    function autoResizeTextarea() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    }

    // Handle Escape key
    function handleEscapeKey(event) {
        if (event.key === 'Escape' && emojiPicker.style.display === 'block') {
            emojiPicker.style.display = 'none';
        }
    }

    // Send message to server
    function sendMessage() {
        const message = userInput.value.trim();

        if (message === '') {
            appendMessage('⚠ Please enter a message!', 'bot');
            return;
        }

        document.querySelector('.welcome-message').style.display = 'none';
        appendMessage(message, 'user');

        userInput.value = '';
        userInput.style.height = 'auto';
        userInput.focus();

        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message bot-message typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        document.getElementById('chat-box').appendChild(typingIndicator);
        document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;

        // Send to server
        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            document.querySelectorAll('.typing-indicator').forEach(ind => ind.remove());
            
            if (data.response) {
                appendMessage(formatResponse(data.response), 'bot');
                fetchChatSessions(); // Refresh session list
            } else {
                appendMessage("⚠ Error: Could not fetch response.", 'bot');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.querySelectorAll('.typing-indicator').forEach(ind => ind.remove());
            appendMessage("⚠ Error: Unable to connect to the server.", 'bot');
        });
    }

    function formatResponse(response) {
        return response
            .replace(/\\(.+?)\\/g, '🔹 $1')     // \text\ → 🔹 text
            .replace(/\n/g, '<br>');           // line breaks → <br>
    }
    
    

    // Append message to chat
    function appendMessage(message, sender) {
        const chatBox = document.getElementById('chat-box');
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender + '-message');

        const messageContent = document.createElement('span');
        messageContent.innerHTML = message;

        messageDiv.appendChild(messageContent);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Handle file upload
    function handleFileUpload() {
        const file = fileUpload.files[0];
        const instructions = prompt("What would you like me to do with this file? (e.g., summarize, extract key points)") || "Analyze this document";

        if (!file) {
            appendMessage("⚠ No file selected!", 'bot');
            return;
        }

        appendMessage(`📂 Uploading file: ${file.name}...`, 'user');
        loadingIndicator.style.display = 'flex';

        const formData = new FormData();
        formData.append("file", file);
        formData.append("instructions", instructions);

        fetch("/api/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            if (data.ai_response) {
                appendMessage(`📄 Analysis: ${data.ai_response}`, 'bot');
            } else if (data.error) {
                appendMessage(`⚠ ${data.error}`, 'bot');
            } else {
                appendMessage("⚠ Unexpected response from server", 'bot');
            }
        })
        .catch(error => {
            console.error("Error uploading file:", error);
            loadingIndicator.style.display = 'none';
            appendMessage("⚠ Error uploading file. Please try again.", 'bot');
        });

        fileUpload.value = "";
    }

    // Toggle speech recognition
    function toggleSpeechRecognition() {
        if (isRecognizing) {
            stopSpeechRecognition();
            return;
        }
        
        startSpeechRecognition();
    }

    // Start speech recognition
    function startSpeechRecognition() {
        // Check if browser supports speech recognition
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            appendMessage("⚠ Your browser doesn't support speech recognition. Try Chrome or Edge.", 'bot');
            return;
        }

        isRecognizing = true;
        speechBtn.classList.add('active');
        appendMessage("🎤 Listening...", 'bot');
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            
            // Automatically send the message
            setTimeout(() => {
                sendMessage();
            }, 500);
        };
        
        recognition.onerror = (event) => {
            console.error("Speech recognition error", event.error);
            appendMessage(`⚠ Error: ${event.error}`, 'bot');
            stopSpeechRecognition();
        };

        recognition.onend = () => {
            stopSpeechRecognition();
        };

        recognition.start();
    }

    // Stop speech recognition
    function stopSpeechRecognition() {
        isRecognizing = false;
        speechBtn.classList.remove('active');
    }

    // Toggle dark mode
    function toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
        
        // Save preference to localStorage
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'enabled');
        } else {
            localStorage.setItem('darkMode', 'disabled');
        }
    }

    // Toggle emoji picker
    function toggleEmojiPicker(e) {
        e.stopPropagation();
        const currentDisplay = emojiPicker.style.display;
        emojiPicker.style.display = currentDisplay === 'block' ? 'none' : 'block';
    }


    // function toggleEmojiPicker() {
    //     // emojiPicker.style.display = (emojiPicker.style.display === 'block') ? 'none' : 'block';

    //     emojiContainer.style.display = (emojiContainer.style.display === 'block') ? 'none' : 'block';

    // }

    // Add emoji to input
    function addEmoji(emoji) {
        userInput.value += emoji;
        userInput.focus();
        // emojiPicker.style.display = 'none';
    }

    function sendQuickMessage(topic) {
        let message = '';
        switch(topic) {
            case 'Address':
                message = "What is the address of SIGCE?";
                break;
            case 'Courses':
                message = "What B.E. courses are offered at SIGCE?";
                break;
            case 'Fees':
                message = "What is the fee structure for SIGCE ?";
                break;
            case 'Contact':
                message = "What is the contact information for SIGCE?";
                break;
            default:
                message = `Tell me about ${topic}`;
        }
        
        if (userInput) {
            userInput.value = message;
            userInput.focus(); // Focus the input field
        }
    }

    

    // Initialize the app
    init();
});