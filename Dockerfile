FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
LABEL authors="julian_hoellig"

# install tzdata
RUN apt-get update && apt-get install -y tzdata

# Set timezone to UTC to avoid interactive mode when installing R
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend Noninteractive tzdata

# install python, java, R, and other dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    g++ \
    python3-dev \
    libsndfile1 \
    openjdk-17-jdk-headless \
    r-base \
    && apt-get clean

# check if Python, Java and R work
RUN python3 --version && \
    java -version && \
    R --version \

WORKDIR /app
RUN apt install python3-pip -y
COPY requirements.txt eval_analysis.py IEEE_pipeline.py pps.py /app/
RUN Rscript -e "install.packages(c('data.table', 'ranger', 'foreach', 'truncnorm', 'matrixStats', 'arf'), repos='https://cloud.r-project.org')"
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY methods /app/methods/
COPY data /app/data/
COPY eval /app/eval/
RUN apt-get update && apt-get install -y wget