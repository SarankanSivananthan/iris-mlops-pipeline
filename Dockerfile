FROM python:3.9.7

#Set the working directory to /app
WORKDIR /app

#Copy the current directory contents
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

#Make port 5000 available to the world outside this container
EXPOSE 5000

#Run get_model.py when the container launches
CMD python get_model.py
