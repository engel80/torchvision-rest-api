# FROM ubuntu uncomment if you want to use ubuntu

# Install pip and git
# RUN apt-get update && apt-get install --assume-yes --fix-missing python-pip git

# Clone repository to /app folder in the container image
# RUN git clone https://github.com/deepakiim/Deploy-machine-learning-model.git /app

#####################################################################################################################
# FROM python:3.6.6-slim
# FROM python:3.8.13-slim-buster
FROM anibali/pytorch:1.10.2-cuda11.3

# Mount current directory to /app in the container image
VOLUME ./:app/

RUN apt update && \
    apt install htop

COPY requirements.txt requirements.txt
# Install dependencies
# use --proxy http://<proxy host>:port if you have proxy
RUN pip install -r requirements.txt

COPY . /app/
WORKDIR /app
EXPOSE 8000

# ENV FLASK_ENV=development 
ENV FLASK_APP=app.py
 
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
# CMD ["flask", "run"]