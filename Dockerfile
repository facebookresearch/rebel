FROM continuumio/miniconda3:4.8.2

RUN mkdir /code

WORKDIR /code

#RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list && \
RUN	  sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list && \
	apt-get update --allow-releaseinfo-change && \
	apt-get install -y --allow-unauthenticated \
	apt-utils \
	ca-certificates \
	curl \
	gnupg \
	git build-essential \
	nvidia-cuda-toolkit \
	--no-install-recommends && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/facebookresearch/rebel.git
WORKDIR /code/rebel

RUN	conda create --yes -n rebel python=3.7 && \
	conda init bash && echo "source activate rebel" > ~/.bashrc
ENV PATH /opt/conda/envs/rebel/bin:$PATH

RUN pip install --no-cache-dir -r requirements.txt && \
	conda install cmake
RUN pip uninstall -y torch==1.4.0
RUN pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN make


