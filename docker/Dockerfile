#FROM ubuntu:23.10
FROM python:3.9.17

COPY . /digits/

WORKDIR /digits
RUN pip3 install -r requirements.txt
# WORKDIR /digits

# Export flask
ENV FLASK_APP="api/app"
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD ["python", "api/app.py"]