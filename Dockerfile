FROM kaixhin/cuda-keras
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


# set keras backend to theano
ENV KERAS_BACKEND=theano
ENV BASE_PATH="/training_files"