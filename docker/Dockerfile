FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps:
# - git: clone rosbagkit
# - libgl1/libglib2.0-0: common runtime deps for opencv-python
# - bash/ca-certificates: convenience and HTTPS certs
# - vim/nano included for editing yaml files
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    vim \
    nano \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=rosbagkit
ARG USER_UID=1000
ARG USER_GID=1000

# Create or reuse a group/user for the requested UID/GID.
#
# This avoids build failures when:
# - USER_GID already exists in the base image
# - USER_UID already exists in the base image
#
# If the UID already exists, we rename that user to ${USERNAME}.
# If the GID already exists, we reuse that group.
RUN set -eux; \
    if getent group "${USER_GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        GROUP_NAME="${USERNAME}"; \
        groupadd --gid "${USER_GID}" "${GROUP_NAME}"; \
    fi; \
    if getent passwd "${USER_UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        if [ "${EXISTING_USER}" != "${USERNAME}" ]; then \
            usermod -l "${USERNAME}" "${EXISTING_USER}"; \
            usermod -d "/home/${USERNAME}" -m "${USERNAME}" || true; \
        fi; \
        usermod -g "${USER_GID}" "${USERNAME}"; \
        usermod -s /bin/bash "${USERNAME}"; \
    else \
        useradd \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --create-home \
            --shell /bin/bash \
            "${USERNAME}"; \
    fi

WORKDIR /workspace

ARG ROSBAGKIT_REPO=https://github.com/ut-amrl/rosbagkit.git
ARG ROSBAGKIT_COMMIT=main

RUN git clone "${ROSBAGKIT_REPO}" /workspace/rosbagkit \
    && cd /workspace/rosbagkit \
    && git checkout "${ROSBAGKIT_COMMIT}"

WORKDIR /workspace/rosbagkit

# Install Python package dependencies from pyproject.toml.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# External data mount point.
RUN mkdir -p /workspace/data \
    && chown -R "${USER_UID}:${USER_GID}" /workspace

USER ${USERNAME}

CMD ["/bin/bash"]
