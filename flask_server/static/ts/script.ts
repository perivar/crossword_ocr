let has_camera = null;
$(function () {
  // hide buttons
  $("#retry").hide();
  $("#results").hide();

  has_camera = hasCamera();

  if (has_camera) {
    document.getElementById("pc_fileupload").style.display = "none";

    const submit = document.getElementById('submit') as HTMLInputElement | null;
    if (submit != null) {
      submit.value = "capture and text extraction";
    }

    initCamera();
  } else {
    document.getElementById("mob_camera").style.display = "none";
  }

  // event handler for form submission
  setSubmitEvent();
  setRetryEvent();
});

const loadFile = function (event) {
  const image = document.getElementById('image') as HTMLInputElement | null;

  if (image != null) {
    image.src = URL.createObjectURL(event.target.files[0]);
  }
};

// https://discuss.dizzycoding.com/how-do-i-take-picture-from-client-sidehtml-and-save-it-to-server-sidepython/
function setSubmitEvent() {
  $("#submit").on("click", function (event) {
    $("#results").hide();
    const data = new FormData();

    if (has_camera) {
      const captureFile = getCaptureImg();
      data.append("image", captureFile);
    } else {
      const fileInput = $("#file")[0] as HTMLInputElement | null;
      if (fileInput != null) {
        const uploadFile = fileInput.files[0];
        data.append("image", uploadFile);
      }
    }

    $.ajax({
      type: "POST",
      url: "/v1/ocr",
      enctype: "multipart/form-data",
      data: data,
      processData: false,
      contentType: false,
      cache: false,
      timeout: 600000,
      success: function (result) {
        console.log(result);
        $("#post-form").hide();
        $("#retry").show();
        $("#results").show();
        $("#results-data").html(
          "<div class='well'>" + result["output"] + "</div>"
        );
      },
      error: function (error) {
        console.log(error);
      },
    });
  });
}

function setRetryEvent() {
  $("#retry").on("click", function () {
    $("input").val("").show();
    $("#post-form").show();
    $("#retry").hide();
    $("#results").html("");
  });
}

////////////////////////////////////////////////////
// Capture photo and convert to Blob
////////////////////////////////////////////////////

// Set constraints for the video stream
let cameraView: HTMLVideoElement = null;
let cameraCapture: HTMLCanvasElement = null;

async function hasCamera() {

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.log("getUserMedia() not supported. No cameras exist.");
    return false;
  }

  try {
    await navigator.mediaDevices.getUserMedia({
      video: true
    });
    console.log("Successfully found a camera!");
    return true;
  } catch (error) {
    console.error("The browser has no access to a camera.");
    return false;
  }
}

async function initCamera() {
  cameraView = document.querySelector("#camera--view");
  cameraCapture = document.querySelector("#camera--capture");
  await cameraStart();
}

// Access the device camera and stream to cameraView
async function cameraStart() {

  // use rear camera or fallback to other camera if exist (i.e. don't use exact facingMode)
  const constraints = { video: { facingMode: "environment" }, audio: false };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    cameraView.srcObject = stream;
    console.log("Successfully opened camera stream!");
  } catch (error) {
    console.error("The browser has no access to a camera.");
  }
}

function captureCameraImage() {
  cameraCapture.width = cameraView.videoWidth;
  cameraCapture.height = cameraView.videoHeight;
  cameraCapture.getContext("2d").drawImage(cameraView, 0, 0);

  return cameraCapture;
}

//https://samanoske.tistory.com/94
function extractImage(canvas) {
  const imgDataUrl = canvas.toDataURL("image/jpeg");

  const blobBin = atob(imgDataUrl.split(",")[1]); // base64 data decoding
  const array = [];
  for (let i = 0; i < blobBin.length; i++) {
    array.push(blobBin.charCodeAt(i));
  }

  const file = new Blob([new Uint8Array(array)], { type: "image/jpeg" }); // Blob produce
  return file;
}

function getCaptureImg() {
  const canvas = captureCameraImage();
  const cFile = extractImage(canvas);
  return cFile;
}
////////////////////////////////////////////////////////
//end Capture photo and convert to Blob
////////////////////////////////////////////////////////
