FROM nvcr.io/nvidia/tensorflow:21.10-tf1-py3
WORKDIR /home/fyb
COPY . .
COPY pip.conf /root/.config/pip/pip.conf
RUN pip install -r requirements.txt --default-timeout=120
RUN chmod +x /home/fyb/run.sh
ENTRYPOINT ["./run.sh"]
CMD ["python"]