# Use an official Python runtime as a parent image
FROM python:3.10.12

# Set the working directory to /app
WORKDIR /home/gmarinos/Documents/Code/threshold_exceedance_forecasting

# Copy project directory (replace with actual directory)
COPY . /home/gmarinos/Documents/Code/threshold_exceedance_forecasting

RUN PATH="/home/gmarinos/Documents/Code/threshold_exceedance_forecasting/"

# Install pip (if unsure about the base image)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libopenblas-dev libatlas-base-dev liblapack-dev

# Install required packages
RUN pip install --no-cache-dir \
    scikit-learn \
    scikit-survival \
    tsfel \
    tabulate \
    pyyaml \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    requests \
    lightgbm  # Optional, uncomment if needed
 

# Run app.py when the container launches
CMD ["python3", "main.py"]