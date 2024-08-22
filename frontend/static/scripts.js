document.addEventListener("DOMContentLoaded", () => {
  const realtimeRadio = document.querySelector('input[value="realtime"]');
  const videoRadio = document.querySelector('input[value="video"]');
  const fileUploadSection = document.getElementById("file-upload");

  realtimeRadio.addEventListener("change", () => {
    fileUploadSection.style.display = "none";
  });

  videoRadio.addEventListener("change", () => {
    fileUploadSection.style.display = "block";
  });

  const addPhotoButton = document.getElementById("addPhotoButton");
  addPhotoButton.addEventListener("click", () => {
    // Functionality for adding a photo to the database
    // This could open a file dialog or similar functionality
    alert("Functionality to add photo will be implemented here.");
  });

  const uploadButton = document.getElementById("uploadButton");
  uploadButton.addEventListener("click", () => {
    const fileInput = document.getElementById("fileInput");
    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      // Handle file upload and processing here
      alert(`Uploading file: ${file.name}`);
    } else {
      alert("No file selected.");
    }
  });
});
