FROM ghcr.io/james-lg/cyberbattlesim:1 as cb

# Install cyberbattlesim
RUN cp -r /root /cyberbattlesim
RUN chmod 777 /cyberbattlesim
RUN pip install -e /cyberbattlesim

# Development dependencies
ARG USERNAME=vscode
RUN useradd -ms /bin/bash ${USERNAME}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3-pyqt5 x11-apps zsh
RUN pip install pylint
COPY .devcontainer/bashsetup.sh /tmp/library-scripts/
RUN bash /tmp/library-scripts/bashsetup.sh "${USERNAME}"

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

USER ${USERNAME}
WORKDIR /home/${USERNAME}
