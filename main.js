import { KNNImageClassifier } from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';

const IMAGE_SIZE = 227;
const TOPK = 10;
const confidenceThreshold = 0.98;
var words = ["start", "stop"];

class Main {
  constructor() {
    this.exampleCountDisplay = [];
    this.checkMarks = [];
    this.gestureCards = [];
    this.training = -1;
    this.videoPlaying = false;
    this.previousPrediction = -1;
    this.currentPredictedWords = [];
    this.now;
    this.then = Date.now();
    this.startTime = this.then;
    this.fps = 5;
    this.fpsInterval = 1000 / this.fps;
    this.elapsed = 0;
    this.knn = null;
    this.previousKnn = this.knn;
    this.welcomeContainer = document.getElementById("welcomeContainer");
    this.proceedBtn = document.getElementById("proceedButton");
    this.proceedBtn.style.display = "block";
    this.proceedBtn.classList.add("animated");
    this.proceedBtn.classList.add("flash");
    this.proceedBtn.addEventListener('click', () => {
      this.welcomeContainer.classList.add("slideOutUp");
    });
    this.stageTitle = document.getElementById("stage");
    this.stageInstruction = document.getElementById("steps");
    this.predButton = document.getElementById("predictButton");
    this.backToTrainButton = document.getElementById("backButton");
    this.nextButton = document.getElementById('nextButton');
    this.statusContainer = document.getElementById("status");
    this.statusText = document.getElementById("status-text");
    this.translationHolder = document.getElementById("translationHolder");
    this.translationText = document.getElementById("translationText");
    this.translatedCard = document.getElementById("translatedCard");
    this.initialTrainingHolder = document.getElementById('initialTrainingHolder');
    this.videoContainer = document.getElementById("videoHolder");
    this.video = document.getElementById("video");
    this.trainingContainer = document.getElementById("trainingHolder");
    this.addGestureTitle = document.getElementById("add-gesture");
    this.plusImage = document.getElementById("plus_sign");
    this.addWordForm = document.getElementById("add-word");
    this.newWordInput = document.getElementById("new-word");
    this.doneRetrain = document.getElementById("doneRetrain");
    this.trainingCommands = document.getElementById("trainingCommands");
    this.videoCallBtn = document.getElementById("videoCallBtn");
    this.videoCall = document.getElementById("videoCall");
    this.trainedCardsHolder = document.getElementById("trainedCardsHolder");
    this.initializeTranslator();
    this.predictionOutput = new PredictionOutput();
  }

  initializeTranslator() {
    this.startWebcam();
    this.initialTraining();
    this.loadKNN();
  }

  startWebcam() {
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false
    }).then((stream) => {
      this.video.srcObject = stream;
      this.video.width = IMAGE_SIZE;
      this.video.height = IMAGE_SIZE;
      this.video.addEventListener('playing', () => this.videoPlaying = true);
      this.video.addEventListener('paused', () => this.videoPlaying = false);
    });
  }

  initialTraining() {
    this.nextButton.addEventListener('click', () => {
      const exampleCount = this.knn.getClassExampleCount();
      if (Math.max(...exampleCount) > 0) {
        if (exampleCount[0] == 0) {
          alert('Add examples for Start Gesture');
          return;
        }
        if (exampleCount[1] == 0) {
          alert('Add examples for Stop Gesture');
          return;
        }
        this.nextButton.style.display = "none";
        this.stageTitle.innerText = "Continue Training";
        this.stageInstruction.innerText = "Add Gesture Name and Train.";
        this.setupTrainingUI();
      }
    });
    this.initialGestures(0, "startButton");
    this.initialGestures(1, "stopButton");
  }

  loadKNN() {
    this.knn = new KNNImageClassifier(words.length, TOPK);
    this.knn.load().then(() => this.initializeTraining());
  }

  initialGestures(i, btnType) {
    var trainBtn = document.getElementById(btnType);
    trainBtn.addEventListener('click', () => this.train(i));
    var clearBtn = document.getElementById('clear_' + btnType);
    clearBtn.addEventListener('click', () => {
      this.knn.clearClass(i);
      this.exampleCountDisplay[i].innerText = " 0 examples";
      this.gestureCards[i].removeChild(this.gestureCards[i].childNodes[1]);
      this.checkMarks[i].src = "Images/loader.gif";
    });
    var exampleCountDisplay = document.getElementById('counter_' + btnType);
    var checkMark = document.getElementById('checkmark_' + btnType);
    var gestureCard = document.createElement("div");
    gestureCard.className = "trained-gestures";
    var gestureName = document.createElement("h5");
    gestureName.innerText = i == 0 ? "Start" : "Stop";
    gestureCard.appendChild(gestureName);
    this.trainedCardsHolder.appendChild(gestureCard);
    exampleCountDisplay.innerText = " 0 examples";
    checkMark.src = 'Images/loader.gif';
    this.exampleCountDisplay.push(exampleCountDisplay);
    this.checkMarks.push(checkMark);
    this.gestureCards.push(gestureCard);
  }

  setupTrainingUI() {
    const exampleCount = this.knn.getClassExampleCount();
    if (Math.max(...exampleCount) > 0) {
      this.initialTrainingHolder.style.display = "none";
      this.trainingContainer.style.display = "block";
      this.trainedCardsHolder.style.display = "block";
      this.addWordForm.addEventListener('submit', (e) => {
        this.trainingCommands.innerHTML = "";
        e.preventDefault();
        var word = this.newWordInput.value.trim();
        if (word && !words.includes(word)) {
          words.push(word);
          this.createTrainingBtns(words.indexOf(word));
          this.newWordInput.value = '';
          this.knn.numClasses += 1;
          this.knn.classLogitsMatrices.push(null);
          this.knn.classExampleCount.push(0);
          this.initializeTraining();
          this.createTranslateBtn();
        } else {
          alert("Duplicate or invalid word");
        }
      });
    } else {
      alert('Add some examples before proceeding.');
    }
  }

  createTrainingBtns(i) {
    var trainBtn = document.createElement('button');
    trainBtn.className = "trainBtn";
    trainBtn.innerText = "Train";
    this.trainingCommands.appendChild(trainBtn);
    var clearBtn = document.createElement('button');
    clearBtn.className = "clearButton";
    clearBtn.innerText = "Clear";
    this.trainingCommands.appendChild(clearBtn);
    trainBtn.addEventListener('mousedown', () => this.train(i));
    clearBtn.addEventListener('click', () => {
      this.knn.clearClass(i);
      this.exampleCountDisplay[i].innerText = " 0 examples";
      this.gestureCards[i].removeChild(this.gestureCards[i].childNodes[1]);
      this.checkMarks[i].src = 'Images/loader.gif';
    });
    var exampleCountDisplay = document.createElement('h3');
    exampleCountDisplay.style.color = "black";
    this.trainingCommands.appendChild(exampleCountDisplay);
    var checkMark = document.createElement('img');
    checkMark.className = "checkMark";
    this.trainingCommands.appendChild(checkMark);
    var gestureCard = document.createElement("div");
    gestureCard.className = "trained-gestures";
    var gestureName = document.createElement("h5");
    gestureName.innerText = words[i];
    gestureCard.appendChild(gestureName);
    this.trainedCardsHolder.appendChild(gestureCard);
    exampleCountDisplay.innerText = " 0 examples";
    checkMark.src = 'Images/loader.gif';
    this.exampleCountDisplay.push(exampleCountDisplay);
    this.checkMarks.push(checkMark);
    this.gestureCards.push(gestureCard);
  }

  initializeTraining() {
    if (this.timer) this.stopTraining();
    this.video.play();
  }

  train(gestureIndex) {
    if (this.videoPlaying) {
      const image = dl.fromPixels(this.video);
      this.knn.addImage(image, gestureIndex);
      const exampleCount = this.knn.getClassExampleCount()[gestureIndex];
      if (exampleCount > 0) {
        this.exampleCountDisplay[gestureIndex].innerText = ' ' + exampleCount + ' examples';
        if (exampleCount == 1 && !this.gestureCards[gestureIndex].childNodes[1]) {
          var gestureImg = document.createElement("canvas");
          gestureImg.className = "trained_image";
          gestureImg.getContext('2d').drawImage(video, 0, 0, 400, 180);
          this.gestureCards[gestureIndex].appendChild(gestureImg);
        }
        if (exampleCount == 30) {
          this.checkMarks[gestureIndex].src = "Images/checkmark.svg";
        }
      }
    }
  }

  createTranslateBtn() {
    this.predButton.style.display = "block";
    this.predButton.addEventListener('click', () => {
      const exampleCount = this.knn.getClassExampleCount();
      if (Math.max(...exampleCount) > 0) {
        this.video.className = "videoPredict";
        this.stageTitle.innerText = "Translate";
        this.stageInstruction.innerText = "Start Translating.";
        this.trainingContainer.style.display = "none";
        this.translationHolder.style.display = "block";
        this.predButton.style.display = "none";
        this.setUpTranslation();
      } else {
        alert('Add examples before translating.');
      }
    });
  }

  setUpTranslation() {
    if (this.timer) this.stopTraining();
    this.setStatusText("Status: Ready to Predict!", "predict");
    this.video.play();
    this.pred = requestAnimationFrame(this.predict.bind(this));
  }

  predict() {
    this.now = Date.now();
    this.elapsed = this.now - this.then;
    if (this.elapsed > this.fpsInterval) {
      this.then = this.now - this.elapsed % this.fpsInterval;
      if (this.videoPlaying) {
        const exampleCount = this.knn.getClassExampleCount();
        const image = dl.fromPixels(this.video);
        if (Math.max(...exampleCount) > 0) {
          this.knn.predictClass(image).then((res) => {
            for (let i = 0; i < words.length; i++) {
              if (res.classIndex == i && res.confidences[i] > confidenceThreshold && res.classIndex != this.previousPrediction) {
                this.setStatusText("Status: Predicting!", "predict");
                this.predictionOutput.textOutput(words[i], this.gestureCards[i], res.confidences[i] * 100);
                this.previousPrediction = res.classIndex;
              }
            }
          }).then(() => image.dispose());
        } else image.dispose();
      }
    }
    this.pred = requestAnimationFrame(this.predict.bind(this));
  }

  pausePredicting() {
    this.setStatusText("Status: Paused Predicting", "predict");
    cancelAnimationFrame(this.pred);
  }

  stopTraining() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  setStatusText(status) {
    this.statusContainer.style.display = "block";
    this.statusText.innerText = status;
    this.statusContainer.style.backgroundColor = "black";
  }
}

class PredictionOutput {
  constructor() {
    this.synth = window.speechSynthesis;
    this.voices = [];
    this.pitch = 1.0;
    this.rate = 0.9;
    this.translationText = document.getElementById("translationText");
    this.populateVoiceList();
  }

  populateVoiceList() {
    if (typeof speechSynthesis === 'undefined') return;
    this.voices = this.synth.getVoices();
  }

  textOutput(word, gestureCard, confidence) {
    this.translationText.innerText = word + " (" + confidence.toFixed(1) + "%)";
    const utter = new SpeechSynthesisUtterance(word);
    utter.rate = this.rate;
    utter.pitch = this.pitch;
    if (this.voices.length > 0) utter.voice = this.voices[0];
    this.synth.speak(utter);
  }
}

const main = new Main();
