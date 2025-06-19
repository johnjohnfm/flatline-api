const fileInput = document.getElementById("file-input");
const fileNameDisplay = document.getElementById("file-name");
const button = document.getElementById("neutralize-btn");
const processingBox = document.getElementById("processing-box");
const fillBar = document.querySelector(".progress-fill");

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    fileNameDisplay.textContent = fileInput.files[0].name;
  }
});

button.addEventListener("click", async () => {
  if (fileInput.files.length === 0) {
    alert("Please select a .wav file to neutralize.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  processingBox.style.display = "block";
  fillBar.style.width = "0%";
  setTimeout(() => fillBar.style.width = "100%", 100);

  try {
    const response = await fetch("https://flatline-api.onrender.com/neutralize/", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Processing failed.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "flatlined.wav";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    fileNameDisplay.textContent = "File processed and downloaded.";
    fillBar.style.width = "100%";

  } catch (error) {
    alert("Error: " + error.message);
    fillBar.style.width = "0%";
  }
});