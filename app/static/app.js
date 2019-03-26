const recordingDiv = document.getElementById('recording');
const infoDiv = document.getElementById('info');
const actionsDiv = document.getElementById('actions');
const connectionDiv = document.getElementById('connection');

const recordButton = document.getElementById('record');
const submitButton = document.getElementById('classify');

const speciesEl = document.getElementById('species');
const imageEl = document.getElementById('bird_image');
const spect = document.getElementById('spectrogram')

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

function startRecording(button) {
  navigator.mediaDevices.getUserMedia({ audio: true})
    .then(stream => {
      const chunks = [];
      const recorder = new MediaRecorder(stream);
      currentDate = new Date();
      startTime = currentDate.getTime();
      
      button.innerHTML = 'Stop recording';
      submitButton.disabled = true;
      recordingDiv.innerHTML = '';
      infoDiv.innerHTML = '';
      infoDiv.classList.remove('warn');
      connectionDiv.innerHTML = '';
      speciesEl.textContent = '';
      imageEl.hidden = true;

      recorder.ondataavailable = e => {
        console.log('chunk pushed');
        console.log(e.data);
        chunks.push(e.data);
        infoDiv.innerHTML = chunks.length / 10 + ' s'
        // var tempBlob = new Blob(chunks, { type: 'audio/webm' })

        if (recorder.state == 'inactive') {

          const blob = new Blob(chunks, { type: 'audio/webm' });
          button.innerHTML = 'Try again'
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
            infoDiv.classList.add('warn')
            infoDiv.innerHTML = 'This recording is too short, please record at least 5 seconds';
            return 0;
          }

          activateSubmitButton(blob);
          recordButton.classList.remove('primary');
          recordButton.classList.add('secondary');
        }
      }
      recorder.start(100);
      
      button.onclick = function() {
        if (button.innerHTML === 'Stop recording') {
          recorder.stop()
          recordingDiv.innerHTML = ''
        }
        if (button.innerHTML === 'Try again') {
          startRecording(this)
        }
        
      }

    }).catch(console.error);
}


recordButton.onclick = function() {
  startRecording(this)
}