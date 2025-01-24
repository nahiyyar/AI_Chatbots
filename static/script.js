document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chatForm");
    const chatBox = document.getElementById("chat-box");
    const messageInput = document.getElementById("messageInput");

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        // Add user message to chatbox
        appendMessage(userMessage, "user");

        // Clear the input field
        messageInput.value = "";

        try {
            // Fetch bot response
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: userMessage }),
            });

            if (response.ok) {
                const data = await response.json();
                appendMessage(data.response, "bot");
            } else {
                appendMessage("Sorry, I couldn't process your request.", "bot");
            }
        } catch (error) {
            appendMessage("There was an error connecting to the server.", "bot");
        }
    });

    function appendMessage(message, sender) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${sender}-message`;
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
    }
});
