document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const clearButton = document.getElementById('clear-button');
    const resultsContainer = document.getElementById('results-container');
    const detectionImage = document.getElementById('detection-image');
    const detectionList = document.getElementById('detection-list');
    const loading = document.getElementById('loading');

    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropzone.classList.add('highlight');
    }

    function unhighlight(e) {
        dropzone.classList.remove('highlight');
    }

    // Handle file drop
    dropzone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle file input change
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    // Handle file selection
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                displayPreview(file);
                uploadImage(file);
            } else {
                alert('Please select an image file.');
            }
        }
    }

    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.hidden = false;
            dropzone.hidden = true;
        };
        reader.readAsDataURL(file);
    }

    // Clear preview and reset
    clearButton.addEventListener('click', () => {
        previewContainer.hidden = true;
        dropzone.hidden = false;
        resultsContainer.hidden = true;
        previewImage.src = '';
        detectionList.innerHTML = '';
    });

    // Upload image and get predictions
    function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        loading.hidden = false;
        resultsContainer.hidden = true;

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.hidden = true;
            if (data.error) {
                alert(data.error);
                return;
            }
            displayResults(data);
        })
        .catch(error => {
            loading.hidden = true;
            alert('Error processing image. Please try again.');
            console.error('Error:', error);
        });
    }

    // Display detection results
    function displayResults(data) {
        // Display image with detections
        detectionImage.src = data.image;
        
        // Clear previous detections
        detectionList.innerHTML = '';
        
        // Add new detections
        data.detections.forEach(detection => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            
            const name = document.createElement('h3');
            name.textContent = detection.label;
            
            const confidence = document.createElement('p');
            confidence.className = 'confidence';
            confidence.textContent = `Confidence: ${(detection.confidence * 100).toFixed(1)}%`;
            
            detectionItem.appendChild(name);
            detectionItem.appendChild(confidence);
            detectionList.appendChild(detectionItem);
        });
        
        // Show results container
        resultsContainer.hidden = false;
    }
}); 