// Initialize particles.js background
document.addEventListener('DOMContentLoaded', function() {
    particlesJS('particles-js', {
        "particles": {
            "number": {
                "value": 80,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": "#75AADB"
            },
            "shape": {
                "type": "circle",
                "stroke": {
                    "width": 0,
                    "color": "#000000"
                }
            },
            "opacity": {
                "value": 0.5,
                "random": false,
                "anim": {
                    "enable": false,
                    "speed": 1,
                    "opacity_min": 0.1,
                    "sync": false
                }
            },
            "size": {
                "value": 3,
                "random": true,
                "anim": {
                    "enable": false,
                    "speed": 40,
                    "size_min": 0.1,
                    "sync": false
                }
            },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": "#75AADB",
                "opacity": 0.4,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": 2,
                "direction": "none",
                "random": false,
                "straight": false,
                "out_mode": "out",
                "bounce": false
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "grab"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                },
                "resize": true
            },
            "modes": {
                "grab": {
                    "distance": 140,
                    "line_linked": {
                        "opacity": 1
                    }
                },
                "bubble": {
                    "distance": 400,
                    "size": 40,
                    "duration": 2,
                    "opacity": 8,
                    "speed": 3
                },
                "push": {
                    "particles_nb": 4
                }
            }
        },
        "retina_detect": true
    });

    // Initialize tooltips
    $('[data-toggle="tooltip"]').tooltip();

    // Set current year in footer
    $('#currentYear').text(new Date().getFullYear());
});

// Common file upload functionality
function setupFileUpload(uploadBoxId, previewId, clearBtnId, fileInputId) {
    const uploadBox = $(`#${uploadBoxId}`);
    const uploadPreview = $(`#${previewId}`);
    const previewImage = $(`#${previewId} img`);
    const clearPreview = $(`#${clearBtnId}`);
    const fileInput = $(`#${fileInputId}`);

    // Handle drag and drop
    uploadBox.on('dragover', function(e) {
        e.preventDefault();
        $(this).addClass('dragover');
    });

    uploadBox.on('dragleave', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
    });

    uploadBox.on('drop', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
        if (e.originalEvent.dataTransfer.files.length) {
            fileInput[0].files = e.originalEvent.dataTransfer.files;
            displayPreview(fileInput[0].files[0]);
        }
    });

    // Handle click to browse
    uploadBox.on('click', function() {
        fileInput.click();
    });

    // Handle file selection
    fileInput.on('change', function(e) {
        if (e.target.files.length) {
            displayPreview(e.target.files[0]);
        }
    });

    // Handle clear preview
    clearPreview.on('click', function(e) {
        e.stopPropagation();
        fileInput.val('');
        uploadPreview.hide();
        uploadBox.fadeIn();
        $(`#${previewId.replace('Preview', 'Results')}`).hide();
    });

    function displayPreview(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file (JPEG, PNG, etc.)');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(event) {
            previewImage.attr('src', event.target.result);
            uploadBox.hide();
            uploadPreview.fadeIn();
        };
        reader.readAsDataURL(file);
    }
}

// Generate Page Specific Functions
function initGeneratePage() {
    setupFileUpload('uploadBox', 'uploadPreview', 'clearPreview', 'imageUpload');

    // Epsilon slider
    $('#epsilon').on('input', function() {
        $('#epsilonValue').text($(this).val());
    });

    // Generate attack button
    $('#generateBtn').on('click', function() {
        const fileInput = $('#imageUpload')[0];
        if (!fileInput.files.length) {
            alert('Please upload an image first');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('epsilon', $('#epsilon').val());

        $('#loadingOverlay').fadeIn();

        $.ajax({
            url: '/generate',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                $('#originalImage').attr('src', data.original_image);
                $('#adversarialImage').attr('src', data.adversarial_image);
                $('#differenceImage').attr('src', data.difference_image);
                
                $('#originalPrediction').text(data.original_prediction);
                $('#adversarialPrediction').text(data.adversarial_prediction);
                
                $('#results').fadeIn();
                $('#loadingOverlay').fadeOut();
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.error);
                $('#loadingOverlay').fadeOut();
            }
        });
    });
}

// Detect Page Specific Functions
function initDetectPage() {
    setupFileUpload('uploadBox', 'uploadPreview', 'clearPreview', 'imageUpload');

    // Detect attack button
    $('#detectBtn').on('click', function() {
        const fileInput = $('#imageUpload')[0];
        if (!fileInput.files.length) {
            alert('Please upload an image first');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        $('#loadingOverlay').fadeIn();

        $.ajax({
            url: '/detect',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                // Convert boolean to string if needed
                $('#adversarialScore').text(data.adversarial_score.toFixed(2));
                $('#adversarialBar').css('width', (data.adversarial_score * 100) + '%');
                
                // Update verdict
                const verdict = $('#verdict');
                verdict.removeClass('clean suspicious adversarial');
                
                if (data.is_adversarial) {
                    verdict.addClass('adversarial');
                    verdict.find('.verdict-icon').removeClass().addClass('fas fa-exclamation-triangle verdict-icon');
                    verdict.find('.verdict-text').text('Adversarial Attack Detected');
                } else {
                    verdict.addClass('clean');
                    verdict.find('.verdict-icon').removeClass().addClass('fas fa-check-circle verdict-icon');
                    verdict.find('.verdict-text').text('No Adversarial Perturbations Found');
                }
                
                // Update analysis details
                $('#laplacianVar').text(data.laplacian_variance.toFixed(2));
                $('#featureInconsistency').text(data.feature_inconsistency.toFixed(2));
                $('#predictionConfidence').text(data.prediction_confidence.toFixed(2));
                
                // Update visualizations
                $('#frequencyImage').attr('src', data.frequency_analysis);
                $('#gradientImage').attr('src', data.gradient_visualization);
                
                $('#results').fadeIn();
                $('#loadingOverlay').fadeOut();
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.error);
                $('#loadingOverlay').fadeOut();
            }
        });
    });
}

// Defend Page Specific Functions
function initDefendPage() {
    setupFileUpload('uploadBox', 'uploadPreview', 'clearPreview', 'imageUpload');

    // Defend button
    $('#defendBtn').on('click', function() {
        const fileInput = $('#imageUpload')[0];
        if (!fileInput.files.length) {
            alert('Please upload an image first');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('gaussian_blur', $('#gaussianBlur').is(':checked'));
        formData.append('quantization', $('#quantization').is(':checked'));
        formData.append('feature_squeezing', $('#featureSqueezing').is(':checked'));

        $('#loadingOverlay').fadeIn();

        $.ajax({
            url: '/defend',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                // Update images
                $('#originalImage').attr('src', data.original_image);
                $('#adversarialImage').attr('src', data.adversarial_image);
                $('#defendedImage').attr('src', data.defended_image);
                
                // Update predictions
                $('#originalPrediction').text(data.original_prediction);
                $('#adversarialPrediction').text(data.adversarial_prediction);
                $('#defendedPrediction').text(data.defended_prediction);
                
                // Update metrics
                $('#attackSuccessRate').text(data.attack_success_rate + '%');
                $('#defenseSuccessRate').text(data.defense_success_rate + '%');
                $('#psnrValue').text(data.psnr + ' dB');
                
                // Update chart
                updateConfidenceChart(data.confidence_data);
                
                $('#results').fadeIn();
                $('#loadingOverlay').fadeOut();
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.error);
                $('#loadingOverlay').fadeOut();
            }
        });
    });

    function updateConfidenceChart(data) {
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        
        if (window.confidenceChart) {
            window.confidenceChart.destroy();
        }
        
        window.confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Original', 'Adversarial', 'Defended'],
                datasets: [{
                    label: 'Top Prediction Confidence',
                    data: [data.original, data.adversarial, data.defended],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(75, 192, 192, 0.7)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
}

// Initialize page-specific functions based on current page
$(document).ready(function() {
    const path = window.location.pathname;
    
    if (path.endsWith('/generate') || path.endsWith('/generate/')) {
        initGeneratePage();
    } 
    else if (path.endsWith('/detect') || path.endsWith('/detect/')) {
        initDetectPage();
    } 
    else if (path.endsWith('/defend') || path.endsWith('/defend/')) {
        initDefendPage();
    }
    
    // Smooth scrolling for anchor links
    $('a[href^="#"]').on('click', function(event) {
        event.preventDefault();
        $('html, body').animate({
            scrollTop: $($(this).attr('href')).offset().top - 20
        }, 500);
    });
    
    // Mobile menu toggle
    $('#mobileMenuToggle').on('click', function() {
        $('#navLinks').toggleClass('active');
    });
});