# az acr build --file docker/dockerfile --registry mlopswork --image digits_dependencies:latest .
# we can also build and push image using above command
docker build -t digits_dependencies:latest -f docker/Dockerfile .

docker build -t mlopswork.azurecr.io/digits_final:v1 -f docker/Dockerfile .