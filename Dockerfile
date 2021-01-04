FROM python:3.7.4-stretch

RUN mkdir /app
WORKDIR /app

# copy all files
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# run 
CMD ["python","./app.py"]