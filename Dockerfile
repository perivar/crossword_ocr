# syntax=docker/dockerfile:1

FROM ubuntu:22.04

# Turn off debconf messages during build
ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

# Install system dependencies
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:alex-p/tesseract-ocr5
RUN apt-get update && apt-get install -y tesseract-ocr

# Docker says run apt-get update and install together,
# and then rm /var/lib/apt/lists to reduce image size.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-pil \
    python3-requests \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# Add requirements.txt before rest of repo, for caching
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

# Have to put symlinks at the end, since each RUN creates a new SHELL
RUN cd /usr/local/bin \
  && ln -s /usr/bin/python3 python

# ADD . /app
COPY ./flask_server /flask_server
WORKDIR /flask_server

# Remove Click runtime error: Python was configured to use ASCII as encoding for the environment
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 8050

# Set environment variables per https://dev.to/pacheco/dockerize-a-flask-app-and-debug-with-vscode-34i1
ENV FLASK_ENV="development"
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=1

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Run app.py directly - this will run flask according to the parameters in app.run()
# defaults to 5000, unless the parameters have been changed as it has been here to 8050
CMD ["python3", "app.py"]

# Run flask on default port 5000
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

# Run flask on a specified port instead
# CMD [ "python3", "-m" , "flask", "run", "--host" ,"0.0.0.0", "--port", "8050"]


