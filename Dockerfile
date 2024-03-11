# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /home/gmarinos/Documents/Code/threshold_exceedance_forecasting

# Copy the current directory contents into the container at /app
COPY . /home/gmarinos/Documents/Code/threshold_exceedance_forecasting

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install numpy pandas seaborn scikit-learn torch==1.3.1 torchvision==0.4.2 tqdm==4.63.1 #change the libraries

# Make port 80 available to the world outside this container
#change the port  EXPOSE 80  

# Define environment variable
ENV NAME World #change the name

# Run app.py when the container launches
CMD ["python", "put here the link to the main script"]