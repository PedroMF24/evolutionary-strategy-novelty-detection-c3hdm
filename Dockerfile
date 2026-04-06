FROM ubuntu:22.04
RUN apt update
RUN apt install python3-pip gfortran wget cmake git -y
ADD https://feynarts.de/cuba/Cuba-4.2.tar.gz /
ADD https://gitlab.com/higgsbounds/higgstools/-/archive/v1.1.3/higgstools-v1.1.3.tar.gz /
ADD LoopTools-2.16-modified.tar.gz /usr/local/lib
RUN mv /usr/local/lib/LoopTools-2.16 /usr/local/lib/LoopTools-2.16-gfortran
RUN tar xzf Cuba-4.2.tar.gz --directory=/usr/local/lib
RUN tar xzf higgstools-v1.1.3.tar.gz --directory=/usr/local/lib
RUN rm Cuba-4.2.tar.gz higgstools-v1.1.3.tar.gz
RUN cd /usr/local/lib/LoopTools-2.16-gfortran && \
    ./configure --prefix=/usr/local/lib/LoopTools-2.16-gfortran && \
    make && \
    make install
RUN cd /usr/local/lib/Cuba-4.2 && \
    ./configure --prefix=/usr/local/lib/Cuba-4.2 && \
    make && \
    make install
RUN cd /usr/local/lib/higgstools-v1.1.3 && pip install .
ADD https://gitlab.com/higgsbounds/hbdataset/-/archive/v1.6/hbdataset-v1.6.tar.gz /
ADD https://gitlab.com/higgsbounds/hsdataset/-/archive/v1.1/hsdataset-v1.1.tar.gz /
RUN tar xzf hbdataset-v1.6.tar.gz --directory=/usr/local/lib
RUN tar xzf hsdataset-v1.1.tar.gz --directory=/usr/local/lib
RUN rm hbdataset-v1.6.tar.gz hsdataset-v1.1.tar.gz
WORKDIR /app
ADD C3HDMZ2xZ2-AI-1.0.0 /app/
RUN make
ADD requirements.txt /app/
RUN pip install -r requirements.txt
RUN mkdir -p /app/aim
ADD src /app/
ADD configs /app/
RUN touch /app/centroid_seeds.parquet
RUN chmod -R a+wrx /app
CMD [ "python3", "scan.py" ]
