FROM ghcr.io/james-lg/cyberbattlesim:1 as cb

# Copy library scripts to execute
COPY .devcontainer/bashsetup.sh /tmp/library-scripts/

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Install common packages, non-root user
    && bash /tmp/library-scripts/bashsetup.sh "${USERNAME}" "${USER_UID}" "${USER_GID}" "true" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install cyberbattlesim
RUN chmod 777 /root &&\
    mkdir -p /home/vscode/.local/lib/python3.8/site-packages/ &&\
    cp -r /root /home/vscode/.local/lib/python3.8/site-packages/cyberbattle &&\
    chown -R vscode /home/vscode/

# Development dependencies
RUN pip install pylint

# Remove library scripts for final image
RUN rm -rf /tmp/library-scripts

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends vim

ENV FLASK_APP web
ENV FLASK_ENV development
