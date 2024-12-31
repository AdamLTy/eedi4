FROM vllm/vllm-openai:v0.6.4.post1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    tmux \
    lsof \
    git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install jupyter

# user properties
ENV USER_NAME charmq
ENV HOME /home/${USER_NAME}
ENV PATH $HOME/.local/bin:$PATH
USER ${USERNAME}

WORKDIR $HOME
