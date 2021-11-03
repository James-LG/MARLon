FROM ghcr.io/james-lg/cyberbattlesim:1 as cb

# Install cyberbattlesim
RUN cp -r /root /cyberbattlesim
RUN chmod 777 /cyberbattlesim
RUN pip install -e /cyberbattlesim

# Development dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3-pyqt5 x11-apps zsh
RUN pip install pylint
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /home/vscode
ENV GIT_EDITOR "code --wait"
