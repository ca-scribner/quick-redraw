FROM python:3.7-buster

RUN pip install --upgrade pip

# WORKDIR /app

# Install longstanding requirements directly first.  
# This repeats things in the requirements.txt, but it lets us cache docker files for quicker
# rebuilds.  If something changes in requirements.txt and we don't cover it here, we'll still
# catch it below
# requirements.txt
RUN pip install sqlalchemy
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install Pillow
RUN pip install tensorflow
RUN pip install numpy
RUN pip install mlflow
RUN pip install requests
RUN pip install ray
RUN pip install ray[tune]
RUN pip install git+https://github.com/ray-project/tune-sklearn.git
RUN pip install pandas

# requirements_dev.txt
RUN pip install cairocffi
RUN pip install matplotlib
RUN pip install pytest
RUN pip install ipykernel
RUN pip install psutil
RUN pip install openpyxl
RUN pip install joblib
RUN pip install pprint
RUN pip install google-api-python-client
RUN pip install google-auth-oauthlib
RUN pip install cryptography

# Now update to be sure we're right
COPY ./requirements.txt ./
COPY ./requirements_dev.txt ./

RUN pip install -U -r requirements_dev.txt
RUN git clone https://github.com/ca-scribner/quick-redraw.git ./quick-redraw
ENV PYTHONPATH=$PYTHONPATH:/quick-redraw

# If I apt-get here, can this ever be cached?  Just in case, left it till the end
# Needed for ray
RUN apt-get update && \
    apt-get install -y rsync

# CMD ["python", "main.py"]