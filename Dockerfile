FROM kaixhin/cuda-keras:8.0
VOLUME ["/training_files", "/acuhub"]

ARG GIT_USERNAME=devacusense
ARG GIT_PASSWORD=Acusenseisno1!
RUN touch ~/.netrc
RUN echo "machine github.com" >> ~/.netrc
RUN echo "login $GIT_USERNAME" >> ~/.netrc
RUN echo "password $GIT_PASSWORD" >> ~/.netrc
RUN git config --global user.name "Dev Acusense"
RUN git config --global user.email dev@acusense.ai

#RUN git clone https://github.com/Acusense/acuhub.git

WORKDIR /acuhub
RUN pip install -r requirements.txt

# set keras backend to theano
ENV KERAS_BACKEND=theano
ENV BASE_PATH="/training_files"

# Run commands to make code work
RUN sudo apt-get update -y
RUN sudo apt-get install graphviz -y
RUN sudo apt-get install libopencv-dev python-opencv -y
RUN sudo apt-get install python-skimage -y

# Working directory
WORKDIR /training_files
