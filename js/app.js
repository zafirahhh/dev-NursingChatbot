// KKH Nursing Chatbot JS

document.addEventListener('DOMContentLoaded', function () {
  const chatWindow = document.getElementById('chat-window');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');

  function addMessage(sender, text) {
    const msg = document.createElement('div');
    msg.className = sender;
    msg.textContent = text;
    chatWindow.appendChild(msg);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  chatForm.addEventListener('submit', function (e) {
    e.preventDefault();
    const question = userInput.value.trim();
    if (!question) return;
    addMessage('user', question);
    userInput.value = '';
    // Simulate bot response
    setTimeout(() => {
      addMessage('bot', 'Thank you for your question. A nurse will respond soon.');
    }, 600);
  });
});
