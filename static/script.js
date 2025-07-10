const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append('file', file);

    resultDiv.innerHTML = "<div class='loading'>Processing...</div>";

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        resultDiv.innerHTML = "<h2>Top 5 Predictions:</h2>";
        data.predictions.forEach((pred, index) => {
            resultDiv.innerHTML += `<p>${index+1}. <strong>${pred[0]}</strong> - ${(pred[1]*100).toFixed(2)}%</p>`;
        });
    } catch (error) {
        resultDiv.innerHTML = "<p class='error'>An error occurred during prediction. Please try again.</p>";
    }
});