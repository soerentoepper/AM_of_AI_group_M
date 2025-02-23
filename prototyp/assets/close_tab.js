window.addEventListener('unload', function (e) {
    fetch('/shutdown', { method: 'POST' });
});

// Function to close the window
function closeWindow() {
    window.close();
}

// Check for the hidden div that signals quitting
function checkForQuit() {
    var hiddenDiv = document.getElementById('hidden-div');
    if (hiddenDiv && hiddenDiv.textContent === "Quitting") {
        closeWindow();
    } else {
        setTimeout(checkForQuit, 100);  // Check again in 100ms
    }
}

// Start checking when the page loads
window.onload = checkForQuit;
