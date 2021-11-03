FROM ghcr.io/james-lg/cyberbattlesim:1 as cb

ARG USERNAME=vscode
RUN useradd -ms /bin/bash ${USERNAME}

# Install cyberbattlesim
# RUN cp -r /root /cyberbattlesim
# RUN chmod 777 /cyberbattlesim
# RUN pip install -e /cyberbattlesim
RUN chmod 777 /root &&\
    mkdir -p /home/vscode/.local/lib/python3.8/site-packages/ &&\
    cp -r /root /home/vscode/.local/lib/python3.8/site-packages/cyberbattle &&\
    chown -R vscode /home/vscode/

# Development dependencies
RUN pip install pylint
COPY .devcontainer/bashsetup.sh /tmp/library-scripts/
RUN bash /tmp/library-scripts/bashsetup.sh "${USERNAME}"

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV FLASK_APP web
ENV FLASK_ENV development
