const recordingDiv = document.getElementById('recording');
const infoDiv = document.getElementById('info');
const actionsDiv = document.getElementById('actions');
const connectionDiv = document.getElementById('connection');

const recordButton = document.getElementById('record');
const submitButton = document.getElementById('classify');

const speciesEl = document.getElementById('species');
const imageEl = document.getElementById('bird_image');
const spect = document.getElementById('spectrogram');

const recordingAnimation = document.createElement('object');
recordingAnimation.type = 'image/svg+xml';
recordingAnimation.data = '../static/images/recording.svg';
recordingAnimation.width = 50;
recordingAnimation.height = 50;

var currentDate = new Date();
var currentTime = currentDate.getTime();


function showAudioElement(blobUrl) {
  const audioEl = document.createElement('audio');
  audioEl.controls = true;
  const sourceEl = document.createElement('source');
  sourceEl.src = blobUrl;
  sourceEl.type = 'audio/webm';
  audioEl.appendChild(sourceEl);
  recordingDiv.appendChild(audioEl)
};

function activateSubmitButton(rec) {
  submitButton.disabled = false;
  submitButton.hidden = false;
  
  submitButton.onclick = function() {
    var request = new XMLHttpRequest();    
    request.open('POST', '/classify', true);
    request.setRequestHeader('X-File-Name', rec.name);
    request.setRequestHeader('X-File-Size', rec.size);
    request.setRequestHeader('Content-Type', rec.type);

    request.onload = function () {
      var data = JSON.parse(this.response);
      
      if (request.status < 200 || request.status > 400) {
        console.log('error connecting to API')
        connectionDiv.innerHTML = 'The server is not available at the moment'
      }
      
      console.log(data);

      speciesEl.innerHTML = 'Species: ' + data.species;

      imageEl.src = data.image_url;
      imageEl.alt = data.species;
      imageEl.hidden = false;
    }
    
    request.send(rec);
  }
}

if (navigator.mediaDevices) {
  console.log('getUserMedia supported.');
  var constraints = { audio: true };
  var chunks = [];
  
  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    var options = {
      audioBitsPerSecond : 128000,
      mimeType : 'audio/webm'
    }
    var recorder = new MediaRecorder(stream, options);

    // visualize(stream);

    recordButton.onclick = function() {
      
      if (this.innerHTML === 'Stop recording') {
        recorder.stop()
        recordingDiv.innerHTML = ''
      }

      if (this.innerHTML === 'Record' || this.innerHTML === 'Try again') {
        chunks = [];
        recorder.start(500);

        currentDate = new Date();
        startTime = currentDate.getTime();
        
        this.innerHTML = 'Stop recording';
        recordingDiv.innerHTML = '';
        recordingDiv.appendChild(recordingAnimation)
        infoDiv.innerHTML = '';
        infoDiv.classList.remove('warn');
        connectionDiv.innerHTML = '';
        speciesEl.textContent = '';
        imageEl.hidden = true;
      }

    }

    recorder.ondataavailable = e => {
      console.log(recorder.state)
      console.log('chunk pushed');
      console.log(e.data);
      chunks.push(e.data);
      infoDiv.innerHTML = Math.floor(chunks.length / 2) + ':' + (chunks.length % 2) * 5 + '0 s';
      // var tempBlob = new Blob(chunks, { type: 'audio/webm' })
    }

    recorder.onstop = function(e) {

      const blob = new Blob(chunks, { type: 'audio/webm' });
      recordButton.innerHTML = 'Try again';
      showAudioElement(URL.createObjectURL(blob));
      console.log(blob);
      console.log(blob.size);

      currentDate = new Date();
      stopTime = currentDate.getTime();
      durationRecording = (stopTime - startTime) / 1000;
      console.log('finished recording: ');
      console.log('duration recording: ' + durationRecording.toString() + 's');

      if (durationRecording < 5) {
        console.log('recording is too short');
        infoDiv.classList.add('warn');
        infoDiv.innerHTML = 'This recording is too short, please record at least 5 seconds';
        return 0;
      }

      activateSubmitButton(blob);
      recordButton.classList.remove('primary');
      recordButton.classList.add('secondary');
    }

  }).catch(console.error);
}