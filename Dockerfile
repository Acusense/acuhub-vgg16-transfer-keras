FROM kaixhin/cuda-keras
VOLUME ["/training_files", "/acuhub"]

#RUN pip install -r /acuhub/requirements.txt
#RUN sudo easy_install --upgrade numpy
#RUN sudo easy_install --upgrade scipy

# set keras backend to tensorflow
ENV KERAS_BACKEND=tensorflow
ENV BASE_PATH="/training_files"
# can be "/cpu:0", "/gpu:0", etc
ENV TENSORFLOW_DEVICE="/gpu:0"

# Run commands to make code work
RUN sudo apt-get update -y
RUN sudo apt-get install graphviz -y
RUN sudo apt-get install libopencv-dev python-opencv -y
RUN sudo apt-get install python-skimage -y

# Numpy / Scipy reqs
RUN sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose -y

# Import the correct Keras config
COPY keras.json /root/.keras/keras.json

# Working directory
WORKDIR /training_files
