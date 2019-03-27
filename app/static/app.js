const recordingDiv = document.getElementById('recording');
const infoDiv = document.getElementById('info');
const actionsDiv = document.getElementById('actions');
const connectionDiv = document.getElementById('connection');
const responseDiv = document.getElementById('response');

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

    request.onerror = function () {
      console.log('error connecting to API')
      connectionDiv.innerHTML = 'There is a problem with the server. Please try again later'
    }

    request.onload = function () {
      console.log(request.status);

      if (request.status < 200 || request.status > 400) {
        console.log('error connecting to API')
        connectionDiv.innerHTML = 'There is a problem with the server. Please try again later'
      }

      var data = JSON.parse(this.response);

      recordButton.classList.remove('secondary')
      recordButton.classList.add('primary')
      submitButton.hidden = true;
      
      console.log(data);

      speciesEl.innerHTML = 'Species: ' + data.top5_1[0];

      imageEl.src = data.image_url;
      imageEl.alt = data.species;
      imageEl.hidden = false;


      top5_1 = document.createElement('ul')
      var i
      for (i = 0; i < 5; i++) {
        species = document.createElement('li')
        species.innerHTML = data.top5_1[i]
        top5_1.appendChild(species)
      }

      top5_2 = document.createElement('ul')
      var i
      for (i = 0; i < 5; i++) {
        species = document.createElement('li')
        species.innerHTML = data.top5_2[i]
        top5_2.appendChild(species)
      }

      top5_3 = document.createElement('ul')
      var i
      for (i = 0; i < 5; i++) {
        species = document.createElement('li')
        species.innerHTML = data.top5_3[i]
        top5_3.appendChild(species)
      }

      responseDiv.appendChild(top5_1);
      responseDiv.appendChild(top5_2);
      responseDiv.appendChild(top5_3);
    }
    
    request.send(rec);
    console.log(request.status);
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

    try {
      var recorder = new MediaRecorder(stream, options);
    }
    catch(err) {
      connectionDiv.innerHTML = "Your browser doesn't support MediaRecorder, please use Chrome, Firefox or Opera";
      recordButton.disabled= true;
    }

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
        submitButton.hidden = true;
        recordButton.classList.remove('secondary')
        recordButton.classList.add('primary')
        recordingDiv.innerHTML = '';
        recordingDiv.appendChild(recordingAnimation)
        infoDiv.innerHTML = '';
        infoDiv.classList.remove('warn');
        connectionDiv.innerHTML = ''
        speciesEl.innerHTML = '';
        imageEl.hidden = true;
        if (this.innerHTML === 'Try again') {
          responseDiv.removeChild(top5_1)
          responseDiv.removeChild(top5_2)
          responseDiv.removeChild(top5_3)
        }
      }
    }

    recorder.ondataavailable = e => {
      console.log(recorder.state)
      console.log('chunk pushed');
      console.log(e.data);
      chunks.push(e.data);
      infoDiv.innerHTML = Math.floor(chunks.length / 2) + ':' + (chunks.length % 2) * 5 + '0 s';
    }

    recorder.onstop = function(e) {
      console.log(chunks)
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