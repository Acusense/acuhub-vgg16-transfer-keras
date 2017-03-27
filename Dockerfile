FROM kaixhin/cuda-keras

VOLUME ["/home/_training_data", "/home/_inputs", "/home/_outputs", "/home/src"]

#RUN pip install -r /acuhub/requirements.txt
#RUN sudo easy_install --upgrade numpy
#RUN sudo easy_install --upgrade scipy

# set keras backend to tensorflow
ENV KERAS_BACKEND=tensorflow
ENV BASE_PATH="/"
# can be "/cpu:0", "/gpu:0", etc
ENV TENSORFLOW_DEVICE="/gpu:0"

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

# Import the correct Keras config
COPY src/keras.json /root/.keras/keras.json


RUN mkdir -p /home/src

COPY src /home/src

RUN find /home/src/scripts -name "*.sh" -exec chmod +x {} +

# Working directory
WORKDIR /home/src
