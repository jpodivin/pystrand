FROM python:3.7

LABEL author="Jiri Podivin"
LABEL version="1.0"
LABEL description="Pystrand development dockerfile"

ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID pystranddev \
    && useradd --uid $USER_UID --gid $USER_GID -m pystranddev

USER pystranddev:pystranddev

COPY . .

RUN pip install --no-cache-dir -r requirements.txt \
    pylint \
    pytest 
    
#RUN pip install -e .[tests] --progress-bar off
