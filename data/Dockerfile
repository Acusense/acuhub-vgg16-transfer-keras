FROM kaixhin/cuda-keras

CWD ./usr/foo

git clone git@github.com:Acusense/acuhub.git

cd /training_files

# set keras backend to theano
ENV KERAS_BACKEND=theano

CMD python