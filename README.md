# RALF - README

Ralf has two usages: in the frontend container and library for other services. To test the library, build and deploy the docker image locally. A valid HuggingFace token is required to run the test. You will be prompted to enter the token when the docker image is deployed.
```
docker build -t ralf-test .
docker run --rm -ti ralf-test
```