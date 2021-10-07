FROM ghcr.io/james-lg/cyberbattlesim:1 as cb

# Install cyberbattlesim
RUN cp -r /root /cyberbattlesim
RUN chmod 777 /cyberbattlesim
RUN pip install -e /cyberbattlesim

RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /home/vscode
