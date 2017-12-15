FROM andrewosh/binder-base

MAINTAINER Björn Dahlgren <bjodah@gmail.com>

USER root

# This dockerfile is designed to run on binder (mybinder.org)
RUN echo "deb http://ftp.debian.org/debian jessie-backports main" | tee -a /etc/apt/sources.list && \
    apt-get update && \
    apt-get --quiet --assume-yes install curl git g++ libgmp-dev binutils-dev bzip2 make cmake sudo \
    python-dev python-pip libgsl-dev liblapack-dev && \
    apt-get clean && \
    curl -LOs http://computation.llnl.gov/projects/sundials/download/sundials-2.7.0.tar.gz && \ 
    tar xzf sundials-2.7.0.tar.gz && mkdir build/ && cd build/ && \
    cmake -DCMAKE_PREFIX_PATH=/usr/local -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DEXAMPLES_ENABLE=OFF -DEXAMPLES_INSTALL=OFF -DLAPACK_ENABLE=ON \
    ../sundials*/ && make install && cd - && rm -r build/ sundials* && \
    python -m pip install --upgrade pip && \
    curl -LOs http://dl.bintray.com/boostorg/release/1.65.0/source/boost_1_65_0.tar.bz2 && \
    tar xjf boost_*.tar.bz2 && cd boost* && ./bootstrap.sh && ./b2 -j 2 --prefix=/usr/local/ install && cd -

USER main
ENV PYCVODES_LAPACK lapack
RUN echo pyodesys[all] | tee requirements.txt
