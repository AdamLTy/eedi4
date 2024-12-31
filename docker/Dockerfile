FROM nvcr.io/nvidia/pytorch:23.06-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    tmux \
    lsof \
    git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# user properties
ENV USER_NAME charmq
ENV HOME /home/${USER_NAME}
ENV PATH $HOME/.local/bin:$PATH
USER ${USERNAME}

WORKDIR $HOME
