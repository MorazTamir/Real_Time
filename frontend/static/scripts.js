document.addEventListener("DOMContentLoaded", () => {
  // Handle file upload
  document
    .getElementById("uploadForm")
    .addEventListener("submit", async (event) => {
      event.preventDefault();

      const fileInput = document.getElementById("fileInput");
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        alert(result.message);
      } catch (error) {
        console.error("Error:", error);
      }
    });

  // Handle adding multiple photos
  document
    .getElementById("addPhotosForm")
    .addEventListener("submit", async (event) => {
      event.preventDefault();

      const filesInput = document.getElementById("filesInput");
      const formData = new FormData();
      for (const file of filesInput.files) {
        formData.append("files[]", file);
      }

      try {
        const response = await fetch("/add_photos", {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        alert(result.message);
      } catch (error) {
        console.error("Error:", error);
      }
    });
});
