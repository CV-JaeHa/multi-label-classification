FROM ubuntu:20.04

RUN apt-get install -y install A

FROM python:3.7.10-buster

RUN pip install numpy pandas matplotlib jupyterlab tqdm scikit-learn

RUN pip3 install torch torchvision

RUN pip3 install opencv-python efficientnet_pytorch


# EXPOSE 9999

# ADD ./run.sh /tmp/run.sh

# RUN chmod +x /tmp/run.sh

WORKDIR /multi_label_classification

# ENTRYPOINT  ["/tmp/run.sh"]