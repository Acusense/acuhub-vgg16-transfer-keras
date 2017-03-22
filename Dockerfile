FROM kaixhin/cuda-keras

# Volumes for training data, inputs, outputs and src code
VOLUME [
  "/home/_training_data",
  "/home/_inputs",
  "/home/_outputs",
  "/home/src"
]

#RUN pip install -r /acuhub/requirements.txt
#RUN sudo easy_install --upgrade numpy
#RUN sudo easy_install --upgrade scipy

# set keras backend to tensorflow
ENV KERAS_BACKEND=tensorflow
ENV BASE_PATH="/"
ENV TENSORFLOW_DEVICE="/gpu:0"

# Run commands to make code work
RUN sudo apt-get update -y
RUN sudo apt-get install graphviz -y
RUN sudo apt-get install libopencv-dev python-opencv -y
RUN sudo apt-get install python-skimage -y

# Numpy / Scipy reqs
RUN sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose -y

# Import the correct Keras config
COPY src/keras.json /root/.keras/keras.json


RUN mkdir /home/src

COPY src /home/src

RUN find /home/src/scripts -name "*.sh" -exec chmod +x {} +

# Working directory
WORKDIR /home/src
