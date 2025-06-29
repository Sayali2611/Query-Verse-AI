document.addEventListener("DOMContentLoaded", function () {
    const userInput = document.getElementById("user-input");
    const emojiPicker = document.getElementById("emoji-picker");
    const emojiContainer = document.querySelector('.emoji-list');
    const menuBtn = document.getElementById('menuBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const historySidebar = document.getElementById('historySidebar');
    const overlay = document.getElementById('overlay');
    const newChatBtn = document.getElementById('newChatBtn');
    const historyItems = document.getElementById('historyItems');

    // Sample chat history data
    let chatHistory = [
        { id: 1, title: "Admissions process", active: false },
        { id: 2, title: "Course requirements", active: false },
        { id: 3, title: "Fee payment options", active: false },
        { id: 4, title: "Contact information", active: false }
    ];

    let currentChatId = null;

    // Initialize the app
    function init() {
        renderHistoryItems();
        setupEventListeners();
    }

    // Set up event listeners
    function setupEventListeners() {
        // Toggle sidebar
        menuBtn.addEventListener('click', toggleSidebar);
        closeSidebarBtn.addEventListener('click', toggleSidebar);
        overlay.addEventListener('click', toggleSidebar);

        // New chat button
        newChatBtn.addEventListener('click', startNewChat);

        // Send message on button click or Enter key
        document.querySelector('.send-btn').addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        // Emoji picker
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && emojiPicker.style.display === 'block') {
                emojiPicker.style.display = 'none';
            }
        });

        // Load emojis
        const emojiList = ['😀', '😂', '😍', '😎', '😢', '😡', '😜', '🤔', '😇'];
        emojiList.forEach(emoji => {
            const emojiButton = document.createElement('button');
            emojiButton.textContent = emoji;
            emojiButton.onclick = function() {
                addEmoji(emoji);
            };
            emojiContainer.appendChild(emojiButton);
        });
    }

    // Toggle sidebar visibility
    function toggleSidebar() {
        historySidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    }

    // Start a new chat
    function startNewChat() {
        // Generate a new chat ID
        const newId = chatHistory.length > 0 ? Math.max(...chatHistory.map(chat => chat.id)) + 1 : 1;
        
        // Create new chat in history
        const newChat = {
            id: newId,
            title: "New Chat",
            active: true
        };
        
        // Update previous active chat
        chatHistory.forEach(chat => chat.active = false);
        
        // Add new chat to history
        chatHistory.unshift(newChat);
        
        // Update current chat ID
        currentChatId = newId;
        
        // Clear messages
        document.getElementById('chat-box').innerHTML = '';
        
        // Show welcome message
        document.querySelector('.welcome-message').style.display = 'block';
        
        // Update UI
        renderHistoryItems();
        
        // Close sidebar if on mobile
        if (window.innerWidth < 768) {
            toggleSidebar();
        }
    }

    // Render history items in the sidebar
    function renderHistoryItems() {
        historyItems.innerHTML = '';
        
        chatHistory.forEach(chat => {
            const item = document.createElement('div');
            item.className = `history-item ${chat.active ? 'active' : ''}`;
            item.textContent = chat.title;
            item.dataset.chatId = chat.id;
            
            item.addEventListener('click', () => {
                // Set this chat as active
                chatHistory.forEach(c => c.active = false);
                chat.active = true;
                currentChatId = chat.id;
                
                // Update UI
                renderHistoryItems();
                
                // Here you would load the chat messages for this chat
                // For now, we'll just clear and show welcome message
                document.getElementById('chat-box').innerHTML = '';
                document.querySelector('.welcome-message').style.display = 'block';
                
                // Close sidebar if on mobile
                if (window.innerWidth < 768) {
                    toggleSidebar();
                }
            });
            
            historyItems.appendChild(item);
        });
    }

    // Function to send a message
    function sendMessage() {
        const userInputField = document.getElementById('user-input');
        let userInput = userInputField.value.trim();

        if (userInput === '') {
            appendMessage('⚠ Please enter a message!', 'bot');
            return;
        }

        // Hide welcome message when first message is sent
        document.querySelector('.welcome-message').style.display = 'none';

        // Append user message to chat box
        appendMessage(userInput, 'user');

        // Clear input field and refocus
        userInputField.value = '';
        userInputField.style.height = 'auto';
        userInputField.focus();

        // Update chat title if it's the first message
        if (currentChatId) {
            const activeChat = chatHistory.find(chat => chat.id === currentChatId);
            if (activeChat && activeChat.title === "New Chat") {
                const shortMessage = userInput.length > 30 ? userInput.substring(0, 30) + "..." : userInput;
                activeChat.title = shortMessage;
                renderHistoryItems();
            }
        }

        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message bot-message typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        document.getElementById('chat-box').appendChild(typingIndicator);
        document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;

        // Simulate bot response after a delay
        setTimeout(() => {
            // Remove typing indicator
            const indicators = document.querySelectorAll('.typing-indicator');
            indicators.forEach(ind => ind.remove());

            // Add bot response
            const response = generateBotResponse(userInput);
            appendMessage(response, 'bot');
        }, 1500);
    }

    // Function to generate a bot response (simulated)
    function generateBotResponse(input) {
        const responses = {
            "admissions": "Our admissions process is simple! You can apply online through our portal. The deadline for applications is June 30th.",
            "courses": "We offer a wide range of courses in Computer Science, Business, and Engineering. Which field are you interested in?",
            "fees": "Tuition fees vary by program. Undergraduate programs start at $15,000 per year. Would you like more specific information?",
            "contact": "You can reach us at (555) 123-4567 or email info@university.edu. Our office hours are Mon-Fri, 9am-5pm."
        };

        input = input.toLowerCase();
        if (input.includes('admission')) return responses.admissions;
        if (input.includes('course')) return responses.courses;
        if (input.includes('fee') || input.includes('payment')) return responses.fees;
        if (input.includes('contact')) return responses.contact;
        
        return "Thank you for your message! Our team will get back to you shortly with more information.";
    }

    // Function to append messages to the chat box
    function appendMessage(message, sender) {
        const chatBox = document.getElementById('chat-box');
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender + '-message');

        const messageContent = document.createElement('span');
        messageContent.innerHTML = message.replace(/\n/g, '<br>');

        messageDiv.appendChild(messageContent);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to handle file upload
    function uploadFile() {
        let fileInput = document.getElementById("file-upload");
        let file = fileInput.files[0];

        if (!file) {
            appendMessage("⚠ No file selected!", 'bot');
            return;
        }

        // Show file name in chat
        appendMessage(`📂 Uploading file: ${file.name}...`, 'user');

        // Simulate file processing
        setTimeout(() => {
            appendMessage("File uploaded successfully! How would you like me to process it?", 'bot');
        }, 2000);

        // Reset file input
        fileInput.value = "";
    }

    // Function to start speech recognition
    function startSpeechRecognition() {
        appendMessage("🎤 Listening... Speak now!", 'bot');
        setTimeout(() => {
            const randomPhrase = ["Admissions", "Courses", "Fees", "Contact information"][Math.floor(Math.random() * 4)];
            document.getElementById('user-input').value = randomPhrase;
            appendMessage(`I heard: "${randomPhrase}"`, 'bot');
        }, 2000);
    }

    // Function to toggle dark mode
    function toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
        
        // Update sidebar colors in dark mode
        if (document.body.classList.contains('dark-mode')) {
            document.querySelector('.history-sidebar').style.background = '#121212';
        } else {
            document.querySelector('.history-sidebar').style.background = 'linear-gradient(90deg, #1a2844, #2c3e50)';
        }
    }

    // Function to toggle emoji picker
    function toggleEmojiPicker() {
        emojiPicker.style.display = (emojiPicker.style.display === 'block') ? 'none' : 'block';
    }

    // Function to send quick message options
    function sendQuickMessage(option) {
        document.getElementById('user-input').value = option;
        sendMessage();
    }

    // Function to add emojis to chat input
    function addEmoji(emoji) {
        const inputField = document.getElementById('user-input');
        inputField.value += emoji;
        inputField.focus();
        emojiPicker.style.display = 'none';
    }

    // Initialize the app
    init();
});



document.addEventListener("DOMContentLoaded", function () {
    const userInput = document.getElementById("user-input");
    const emojiPicker = document.getElementById("emoji-picker");
    const emojiContainer = document.querySelector('.emoji-list');

    userInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            if (event.shiftKey) {
                return; // ✅ Shift+Enter allows a new line
            } else {
                event.preventDefault(); // ✅ Prevent new line and send message
                sendMessage();
            }
        }
    });

    // 🔹 Close emoji picker when Escape key is pressed
    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && emojiPicker.style.display === "block") {
            emojiPicker.style.display = "none";
        }
    });

    // 🔹 Load emojis dynamically into picker
    const emojiList = ['😀', '😂', '😍', '😎', '😢', '😡', '😜', '🤔', '😇'];
    emojiList.forEach(emoji => {
        const emojiButton = document.createElement('button');
        emojiButton.textContent = emoji;
        emojiButton.onclick = function () {
            addEmoji(emoji);
        };
        emojiContainer.appendChild(emojiButton);
    });
});

// Function to send a message
function sendMessage() {
    const userInputField = document.getElementById('user-input');
    let userInput = userInputField.value.trim();

    if (userInput === '') {
        appendMessage('⚠ Please enter a message!', 'bot');
        return;
    }

    // Append user message to chat box
    appendMessage(userInput, 'user');

    // Clear input field and refocus
    userInputField.value = '';
    userInputField.focus();

    // Send message to Flask backend
    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput }) // 🔹 Send JSON instead of URL encoding
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            let formattedResponse = formatResponse(data.response);
            appendMessage(formattedResponse, 'bot');
        } else {
            appendMessage("⚠ Error: Could not fetch response.", 'bot');
        }
    })
    .catch(error => {
        console.error('❌ Error:', error);
        appendMessage("⚠ Error: Unable to connect to the server.", 'bot');
    });
}

// Function to handle file upload
function uploadFile() {
    let fileInput = document.getElementById("file-upload");
    let file = fileInput.files[0];

    if (!file) {
        appendMessage("⚠ No file selected!", 'bot');
        return;
    }

    // Show file name in chat
    appendMessage(`📂 Uploading file: ${file.name}...`, 'user');

    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let responseMessage = data.extracted_text 
            ? `📝 Extracted Text:<br>${data.extracted_text}` 
            : data.image_analysis 
            ? `📸 Image Analysis: ${data.image_analysis}` 
            : "⚠ No meaningful data extracted from the file.";
        appendMessage(responseMessage, 'bot');
    })
    .catch(error => {
        console.error("❌ Error uploading file:", error);
        appendMessage("⚠ Error uploading file. Please try again.", 'bot');
    });

    // Reset file input
    fileInput.value = "";
}

// Function to format AI response (Markdown-like)
// function formatResponse(response) {
//     return response
//         // .replace(/\n/g, ' ') // Remove unnecessary new lines while preserving spaces
//         // .replace(/\.\s+/g, ".<br>") // Ensure sentences appear in structured paragraphs
//         .replace(/\\(.?)\\/g, '🔹 $1'); // Keep existing Markdown-like formatting
// }

function formatResponse(response) {
    return response
        // .replace(/\n/g, ' ') // Uncomment if you want to remove new lines
        // .replace(/\.\s+/g, ".<br>") // Add line breaks after sentences
        .replace(/\\(.+?)\\/g, '🔹 $1'); // Convert \text\ to 🔹 text
}

// function formatResponse(response) {
//     return response
//         .replace(/\\(.+?)\\/g, '🔹 $1') // Convert \text\ to 🔹 text
//         .replace(/\*/g) // Escape * to prevent Markdown formatting
//         .replace(/_/g, '\\_'); // Escape _ to prevent italic formatting
// }


// Function to append messages to the chat box
function appendMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender + '-message');

    const messageContent = document.createElement('span');
    messageContent.innerHTML = message.replace(/\n/g, '<br>'); // 🔹 Preserve line breaks

    messageDiv.appendChild(messageContent);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Function to start speech recognition
function startSpeechRecognition() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.start();

    recognition.onresult = function (event) {
        const userMessage = event.results[0][0].transcript;
        document.getElementById('user-input').value = userMessage;
        sendMessage();
    };
    
    recognition.onerror = function (event) {
        appendMessage("⚠ Voice recognition error. Please try again.", 'bot');
    };
}

// Function to toggle dark mode
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
}

// Function to toggle emoji picker
function toggleEmojiPicker() {
    const emojiPicker = document.getElementById('emoji-picker');
    emojiPicker.style.display = (emojiPicker.style.display === 'block') ? 'none' : 'block';
}

// Function to send quick message options
function sendQuickMessage(option) {
    document.getElementById('user-input').value = option;
    sendMessage();
}

// Function to add emojis to chat input
function addEmoji(emoji) {
    const inputField = document.getElementById('user-input');
    inputField.value += emoji;
    inputField.focus();
}
