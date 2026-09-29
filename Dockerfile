FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
