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

function setupFileUpload(uploadBoxId, previewId, clearBtnId, fileInputId) {
    const uploadBox = $(`#${uploadBoxId}`);
    const uploadPreview = $(`#${previewId}`);
    const previewImage = $(`#${previewId} img`);
    const clearPreview = $(`#${clearBtnId}`);
    const fileInput = $(`#${fileInputId}`);

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

    uploadBox.on('click', function(e) {
        if (e.target === this) { 
            fileInput[0].click(); 
        }
    });

    fileInput.on('change', function(e) {
        if (e.target.files.length) {
            displayPreview(e.target.files[0]);
        }
    });

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

function initGeneratePage() {
    setupFileUpload('uploadBox', 'uploadPreview', 'clearPreview', 'imageUpload');

    $('#epsilon').on('input', function() {
        $('#epsilonValue').text($(this).val());
    });

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

function initDetectPage() {
    setupFileUpload('uploadBox', 'uploadPreview', 'clearPreview', 'imageUpload');

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
                
                $('#adversarialScore').text(data.adversarial_score.toFixed(2));
                $('#adversarialBar').css('width', (data.adversarial_score * 100) + '%');
                
   
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
      
                $('#laplacianVar').text(data.laplacian_variance.toFixed(2));
                $('#featureInconsistency').text(data.feature_inconsistency.toFixed(2));
                $('#predictionConfidence').text(data.prediction_confidence.toFixed(2));
                
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


function initDefendPage() {
    
    const uploadBox = $('#uploadBox');
    const uploadPreview = $('#uploadPreview');
    const previewImage = $('#previewImage');
    const clearPreview = $('#clearPreview');
    const fileInput = $('#imageUpload');
    const defendBtn = $('#defendBtn');
    
    uploadBox.on('click', function(e) {
        if (e.target === this) { 
            fileInput[0].click();
        }
    });
    
    fileInput.on('change', function(e) {
        if (e.target.files.length) {
            const reader = new FileReader();
            reader.onload = function(event) {
                previewImage.attr('src', event.target.result);
                uploadBox.hide();
                uploadPreview.fadeIn();
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });
    
    clearPreview.on('click', function(e) {
        e.stopPropagation();
        fileInput.val('');
        previewImage.attr('src', '');
        uploadPreview.hide();
        uploadBox.fadeIn();
        $('#results').hide();
    });

    defendBtn.off('click').on('click', function() {
        console.log("Defend button clicked");
        
        if (!fileInput[0].files.length) {
            console.log("No file selected");
            alert('Please upload a perturbed image first');
            return;
        }

        const btn = $(this);
        btn.prop('disabled', true).html(
            '<i class="fas fa-spinner fa-spin"></i> Processing...'
        );

        $('#loadingOverlay').fadeIn();
        $('#results').hide();

        const formData = new FormData();
        formData.append('image', fileInput[0].files[0]);

        $.ajax({
            url: '/defend',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                console.log("Success:", data);
                if (data.success) {
                    $('#perturbedImage').attr('src', data.perturbed_image);
                    $('#defendedImage').attr('src', data.defended_image);
                    
                    // Format predictions
                    function formatPreds(pred) {
                        return Object.entries(pred)
                            .map(([label, prob]) => `${label} (${(prob * 100).toFixed(1)}%)`)
                            .join(', ');
                    }
                    
                    $('#perturbedPred').text(formatPreds(data.perturbed_pred));
                    $('#defendedPred').text(formatPreds(data.defended_pred));
                    
                    $('#results').fadeIn();
                } else {
                    alert('Error: ' + (data.error || 'Processing failed'));
                }
            },
            error: function(xhr) {
                console.error("Error:", xhr.responseText);
                alert('Server error occurred');
            },
            complete: function() {
                btn.prop('disabled', false).html(
                    '<i class="fas fa-shield-alt"></i> Defend Image'
                );
                $('#loadingOverlay').fadeOut();
            }
        });
    });
}

$(document).ready(function() {
    const fileInput = $('#imageUpload'); // debugging line
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
    
    $('a[href^="#"]').on('click', function(event) {
        event.preventDefault();
        $('html, body').animate({
            scrollTop: $($(this).attr('href')).offset().top - 20
        }, 500);
    });
    
    $('#mobileMenuToggle').on('click', function() {
        $('#navLinks').toggleClass('active');
    });
});
