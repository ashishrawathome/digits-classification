FROM digits_dependencies:latest

COPY . /digits/
WORKDIR /digits

# Run test cases
RUN pytest

# Export flask
ENV FLASK_APP="api/app"
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]