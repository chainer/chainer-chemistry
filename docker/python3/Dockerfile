FROM chainer/chainer:v6.1.0-python3

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libboost-dev \
    libboost-python-dev \
    libboost-serialization-dev \
    libboost-iostreams-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libeigen3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG RDKIT_VERSION=Release_2017_09_3
ADD https://github.com/rdkit/rdkit/archive/${RDKIT_VERSION}.tar.gz /

# build & install rdkit
RUN tar xf ${RDKIT_VERSION}.tar.gz && \
    mkdir -p rdkit-${RDKIT_VERSION}/build && \
    cd rdkit-${RDKIT_VERSION}/build && \
    cmake \
    -D RDK_BUILD_SWIG_SUPPORT=OFF \
    -D RDK_BUILD_PYTHON_WRAPPERS=ON \
    -D RDK_BUILD_COMPRESSED_SUPPLIERS=ON \
    -D RDK_BUILD_INCHI_SUPPORT=ON \
    -D RDK_BUILD_AVALON_SUPPORT=ON \
    -D RDK_BUILD_CPP_TESTS=OFF \
    -D RDK_INSTALL_INTREE=OFF \
    -D RDK_INSTALL_STATIC_LIBS=OFF \
    -D PYTHON_EXECUTABLE=/usr/bin/python3.5 \
    -D PYTHON_NUMPY_INCLUDE_PATH=/usr/local/lib/python3.5/dist-packages/numpy/core/include \
    -D PYTHON_INSTDIR=/usr/local/lib/python3.5/dist-packages \
    -D Python_ADDITIONAL_VERSIONS=3.5 \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    .. && \
    make -j $(nproc) && \
    make install

RUN rm ${RDKIT_VERSION}.tar.gz && \
    rm rdkit-${RDKIT_VERSION} -rf

# install chainer-chemistry
# matplotlib >= 3.1 requires upgrade of pip
RUN pip3 install --no-cache-dir matplotlib==3.0 chainer-chemistry

