
FROM python:3.8.7

RUN apt-get clean && apt-get update
RUN apt-get install -y ffmpeg

WORKDIR /home/chirps

RUN pip install librosa
RUN pip install flask
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install matplotlib
RUN pip install python-dotenv

COPY app app
COPY birdsong birdsong

ENV FLASK_APP routes.py

EXPOSE 5000

WORKDIR /home/chirps/app
ENTRYPOINT ["python", "-m", "flask", "run", "--host", "0.0.0.0"]