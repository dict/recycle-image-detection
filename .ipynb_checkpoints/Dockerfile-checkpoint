FROM est_nuxlear:v03

LABEL email='dict@estsoft.com'

RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get install -y tzdata

ENV HOME=/home/

RUN mkdir -p ${HOME}/agc2022/dataset

WORKDIR ${HOME}/agc2022

COPY ./src .

CMD ["python3","main.py"]