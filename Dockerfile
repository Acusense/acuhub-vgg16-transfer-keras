FROM kaixhin/cuda-keras

VOLUME ["/home/_data", "/home/_inputs", "/home/_outputs", "/home/src"]

#RUN pip install -r /acuhub/requirements.txt
#RUN sudo easy_install --upgrade numpy
#RUN sudo easy_install --upgrade scipy

# set keras backend to tensorflow
ENV KERAS_BACKEND=tensorflow
# can be "/cpu:0", "/gpu:0", etc
ENV TENSORFLOW_DEVICE="/gpu:0"

# Setup environment variables
ENV INPUT_DIR=/home/_inputs
ENV OUTPUT_DIR=/home/_outputs
ENV DATA_DIR=/home/_data
ENV SRC_DIR=/home/src

RUN sudo apt-get install python-pip python-dev

# Run commands to make code work
RUN sudo apt-get update -y
RUN sudo apt-get install graphviz -y
RUN sudo apt-get install libopencv-dev python-opencv -y
RUN sudo apt-get install python-skimage -y

# Numpy / Scipy reqs
RUN sudo apt-get install python-numpy
RUN sudo apt-get install python-scipy
RUN sudo apt-get install python-matplotlib
RUN sudo apt-get install ipython -y
RUN sudo apt-get install ipython-notebook -y
RUN sudo apt-get install python-pandas -y
RUN sudo apt-get install python-sympy -y
RUN sudo apt-get install python-nose -y




RUN sudo pip  install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl

# Import the correct Keras config
COPY src/keras.json /root/.keras/keras.json


RUN mkdir -p /home/src

COPY src /home/src

RUN find /home/src/scripts -name "*.sh" -exec chmod +x {} +

# Working directory: this is where unix scripts will run from
WORKDIR /home/src
