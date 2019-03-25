const warning = document.getElementById('warn');
const buttonEl = document.getElementById('classify');
const speciesEl = document.getElementById('species');
const imageEl = document.getElementById('bird_image');
const soundDiv = document.getElementById("soundInfo");
const audio = document.getElementById('myAudio')

function activateButtonElement(rec) {
    buttonEl.disabled = false;
    console.log(buttonEl); 
    
    buttonEl.onclick = function() {
        var request = new XMLHttpRequest();    
        request.open('POST', '/classify', true);
        request.setRequestHeader('X-File-Name', rec.name);
        request.setRequestHeader('X-File-Size', rec.size);
        request.setRequestHeader('Content-Type', rec.type);

        console.log('remove player and show spectrogram instead')
        var spectrogram = document.createElement("img");
        spectrogram.src = '../static/images/spect.png'
        soundDiv.innerHTML = ''
        soundDiv.appendChild(spectrogram);

        request.onload = function () {
            var data = JSON.parse(this.response);
            
            if (request.status < 200 || request.status > 400) {
                console.log('error connecting to API')
            }
            
            console.log(data);

            speciesEl.innerHTML = data.species;

            imageEl.src = data.image_url;
            imageEl.alt = data.species;
        };
        
        request.send(rec);

    }
};

var options = {
    controls: true,
    width: 600,
    height: 300,
    fluid: false,
    plugins: {
        wavesurfer: {
            src: 'live',
            waveColor: '#36393b',
            progressColor: 'black',
            debug: true,
            cursorWidth: 1,
            msDisplayMax: 20,
            hideScrollbar: true
        },
        record: {
            audio: true,
            video: false,
            maxLength: 20,
            debug: true
        }
    }
};
var currentDate = new Date();
var currentTime = currentDate.getTime();

// apply audio workarounds for certain browsers
applyAudioWorkaround();
// create player
var player = videojs('myAudio', options, function() {
    // print version information at startup
    var msg = 'Using video.js ' + videojs.VERSION +
        ' with videojs-record ' + videojs.getPluginVersion('record') +
        ', videojs-wavesurfer ' + videojs.getPluginVersion('wavesurfer') +
        ', wavesurfer.js ' + WaveSurfer.VERSION + ' and recordrtc ' +
        RecordRTC.version;
    videojs.log(msg);
});
// error handling
player.on('deviceError', function() {
    console.log('device error:', player.deviceErrorCode);
});
player.on('error', function(element, error) {
    console.error(error);
});
// user clicked the record button and started recording
player.on('startRecord', function() {
    currentDate = new Date()
    startTime = currentDate.getTime()
    console.log('started recording!');
});
// user completed recording and stream is available
player.on('finishRecord', function() {
    // the blob object contains the recorded data that
    // can be downloaded by the user, stored on server etc.
    currentDate = new Date();
    stopTime = currentDate.getTime();
    durationRecording = (stopTime - startTime) / 1000;
    console.log('finished recording: ');
    console.log('duration recording: ' + durationRecording.toString() + 's');
    if (durationRecording < 5) {
        console.log('recording is too short');
        warning.innerHTML = 'This recording is too short, please record at least 5 seconds';
        return 0;
    }
    warning.innerHTML = '';
    const rec = player.recordedData;
    console.log(rec);
    console.log(rec.size)
    //createButtonElement(URL.createObjectURL(rec));
    activateButtonElement(rec);
});