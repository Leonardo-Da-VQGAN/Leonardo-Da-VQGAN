FROM ghcr.io/khlam/docker-pytorch/pyt:latest
WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["main.py"]