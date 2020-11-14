FROM python:3

LABEL author="Jiri Podivin"
LABEL version="0.1"
LABEL description="Pystrand development dockerfile"

COPY . .

RUN pip install --no-cache-dir -r requirements.txt \
    pylint \
    pytest 
    
#RUN pip install -e .[tests] --progress-bar off
