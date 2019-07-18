



FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

RUN pip install flask gevent requests pillow keras theano emoji h5py numpy scikit-learn text-unidecode pandas
COPY . /app
WORKDIR /app






ENTRYPOINT ["python"]
CMD ["run_keras_server.py"]