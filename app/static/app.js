const recordingDiv = document.getElementById('recording');
const infoDiv = document.getElementById('info');
const actionsDiv = document.getElementById('actions');
const connectionDiv = document.getElementById('connection');
const responseDiv = document.getElementById('response');

const recordButton = document.getElementById('record');
const submitButton = document.getElementById('classify');

const speciesEl = document.getElementById('species');
const wikiLinkEl = document.getElementById('wiki_link');
const imgSourceEl = document.getElementById('img_source');
const imageLinkEl = document.getElementById('bird_link');
const imageEl = document.getElementById('bird_image');
const spect = document.getElementById('spectrogram');

const recordingAnimation = document.createElement('object');
recordingAnimation.type = 'image/svg+xml';
recordingAnimation.data = '../static/images/recording.svg';
recordingAnimation.width = 50;
recordingAnimation.height = 50;

const linkIcon = document.createElement('img');
linkIcon.src = '../static/images/info.png';
linkIcon.width = 16;
linkIcon.height = 16;

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

      speciesEl.innerHTML = 'Sounds like ... ' + data.predictions[0][0];

      imageEl.src = data.predictions[0][1];
      imageEl.alt = data.predictions[0][0];
      wikiLinkEl.href = data.predictions[0][3];
      imageLinkEl.href = data.predictions[0][2];
      imageEl.hidden = false;
      imgSourceEl.hidden = false;

      top5 = document.createElement('ul');
      var i
      for (i = 1; i < 5; i++) {
        species = document.createElement('li');
        console.log(species)
        species.innerHTML = data.predictions[i][0] + " ";
        var link = document.createElement('a')
        link.href = data.predictions[i][3]
        link.target = "_blank"
        link.appendChild(linkIcon.cloneNode(true));
        console.log(link)
        species.appendChild(link)
        console.log(species)
        top5.appendChild(species);
      }

      title = document.createElement('p');
      title.innerHTML = 'But it could also be';
      responseDiv.appendChild(title);
      responseDiv.appendChild(top5);
    }
    
    request.send(rec);
    console.log(request.status);
    speciesEl.innerHTML = 'Sounds like ... ';
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
        imgSourceEl.hidden = true;

        if (this.innerHTML === 'Try again') {
          try {
            responseDiv.removeChild(title)
            responseDiv.removeChild(top5)
          }
          catch(err) {}
        }
        this.innerHTML = 'Stop recording';
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