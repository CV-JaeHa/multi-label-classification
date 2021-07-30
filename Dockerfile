FROM ubuntu:20.04

FROM python:3.7.10-buster

RUN pip install numpy pandas matplotlib jupyterlab

RUN pip3 install torch torchvision

RUN pip3 install opencv-python

EXPOSE 9999

ADD ./run.sh /tmp/run.sh

RUN chmod +x /tmp/run.sh

WORKDIR /multi_label_classification

ENTRYPOINT  ["/tmp/run.sh"]