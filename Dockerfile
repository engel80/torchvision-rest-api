#
#
#

# FROM python:3.8.13-slim-buster
FROM anibali/pytorch:1.10.2-cuda11.3

VOLUME ./:app/

RUN sudo apt update && \
    sudo apt install -y htop watch

COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8000

ENV FLASK_ENV=development
ENV FLASK_APP=app.py
 
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
# CMD ["flask", "run"]