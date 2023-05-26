#specifying a base image for a Docker container.
FROM python:3.8

#run when the Docker container is built.
#It updates the package list for the  OS using the apt-get update command, 
#and then upgrades any installed packages to their latest versions using the apt-get upgrade -y command. 
#The -y flag automatically confirms any prompts to upgrade packages without requiring user interaction

RUN apt-get update && apt-get upgrade -y

#run when the Docker container is built.
# It uses the pip3 command to install or upgrade the AWS CLI 

RUN pip3 --no-cache-dir install --upgrade awscli

RUN pip3 --no-cache-dir install --upgrade pip
#This line of code sets the working directory for the Docker container. 
WORKDIR /app

#copy files from the host system into the Docker container.
COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

# The CMD instruction provides the command and arguments to be executed when the container is run
# The command is python3 and the argument is app.py
# The app.py script will be executed using the python3 interpreter when the container is started
CMD ["python3", "app.py"]