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
    alert("Functionality to add photo will be implemented here.");
  });

  const uploadButton = document.getElementById("uploadButton");
  uploadButton.addEventListener("click", () => {
    const fileInput = document.getElementById("fileInput");
    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      alert(`Uploading file: ${file.name}`);
    } else {
      alert("No file selected.");
    }
  });

  const searchButton = document.getElementById("searchButton");
  const searchBox = document.getElementById("searchBox");
  const searchResult = document.getElementById("searchResult");
  const resultImage = document.getElementById("databaseImage");
  const databaseName = document.getElementById("databaseName");

  searchButton.addEventListener("click", () => {
    const searchQuery = searchBox.value.trim();
    if (searchQuery === "") {
      searchResult.textContent = "";
      searchResult.style.display = "none";
      return;
    }

    // Simulate search
    const found = false; // Replace with actual search logic

    if (found) {
      // Display found image
      resultImage.src = `path/to/found-image.jpg`; // Update with the actual image path
      databaseName.textContent = "Found Name"; // Replace with the name from the search result
      searchResult.textContent = "";
      searchResult.style.display = "none";
    } else {
      // Show "Not found" alert
      resultImage.src = ""; // Clear the image if not found
      databaseName.textContent = "";
      searchResult.textContent = "Not found";
      searchResult.style.display = "block";
    }
  });

  const prevImageButton = document.getElementById("prevImageButton");
  const nextImageButton = document.getElementById("nextImageButton");

  prevImageButton.addEventListener("click", () => {
    // Logic to navigate to the previous image
  });

  nextImageButton.addEventListener("click", () => {
    // Logic to navigate to the next image
  });

  const addPersonButton = document.getElementById("addPersonButton");
  const personNameInput = document.getElementById("personNameInput");

  addPersonButton.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      const personName = personNameInput.value.trim();

      if (file && personName) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("name", personName);

        fetch("/add-person", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.success) {
              alert("Person added successfully.");
            } else {
              alert("Error adding person.");
            }
          })
          .catch((error) => {
            console.error("Error:", error);
          });
      } else {
        alert("Please provide a name and select an image.");
      }
    });

    fileInput.click();
  });
});
