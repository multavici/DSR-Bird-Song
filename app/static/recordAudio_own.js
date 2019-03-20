const recordButton = document.querySelector('#record-button');
const vidjs = require('videojs-record')

var wavesurfer = WaveSurfer.create({
  container: '#waveform',
  waveColor: 'violet',
  progressColor: 'purple',
  backend: 'MediaElement'
});

function createAudioElement(blobUrl) {
  const downloadEl = document.createElement('a');
  downloadEl.innerHTML = 'Identify';
  downloadEl.download = 'audio.webm';
  downloadEl.href = blobUrl;
  const audioEl = document.createElement('audio');
  audioEl.controls = true;
  const sourceEl = document.createElement('source');
  sourceEl.src = blobUrl;
  sourceEl.type = 'audio/webm';
  audioEl.appendChild(sourceEl);
  document.body.appendChild(audioEl);
  document.body.appendChild(downloadEl);
};



recordButton.onclick = function() {
  navigator.mediaDevices.getUserMedia({ audio: true})
    .then(stream => {
      const chunks = [];
      const recorder = new MediaRecorder(stream);
      
      recordButton.innerHTML = 'Stop recording'

      recorder.ondataavailable = e => {
        console.log('chunk pushed');
        console.log(e.data);
        chunks.push(e.data);
        var tempBlob = new Blob(chunks, { type: 'audio/webm' })
        
        wavesurfer.load(tempBlob);

        if (recorder.state == 'inactive') {
          const blob = new Blob(chunks, { type: 'audio/webm' });
          createAudioElement(URL.createObjectURL(blob));
          console.log(blob);
          console.log(blob.size);
        }
      }
      recorder.start(1000);
      
      recordButton.onclick = function() {
        recorder.stop()
        recordButton.innerHTML = 'Try again'
      }

    }).catch(console.error);
}

function handleSuccess(stream) {
  window.stream = stream;
  console.log(stream)
}