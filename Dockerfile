# FROM ubuntu uncomment if you want to use ubuntu

# Install pip and git
# RUN apt-get update && apt-get install --assume-yes --fix-missing python-pip git

# Clone repository to /app folder in the container image
# RUN git clone https://github.com/deepakiim/Deploy-machine-learning-model.git /app

#####################################################################################################################
# FROM python:3.6.6-slim
FROM python:3.8.13-slim-buster

# Mount current directory to /app in the container image
VOLUME ./:app/

COPY requirements.txt requirements.txt
# Install dependencies
# use --proxy http://<proxy host>:port if you have proxy
RUN pip install -r requirements.txt

# Copy local directory to /app in container
# Dont use COPY * /app/ , * will lead to lose of folder structure in /app
COPY . /app/

WORKDIR /app

# In Docker, the containers themselves can have applications running on ports. To access these applications, we need to expose the containers internal port and bind the exposed port to a specified port on the host.
# Expose port and run the application when the container is started
EXPOSE 8000

# ENTRYPOINT python flask_api.py 8000
# CMD ["gunicorn", "flask_api:app", "--bind", "0.0.0.0:8000"]
# ENV CUDA_VISIBLE_DEVICES=0

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]