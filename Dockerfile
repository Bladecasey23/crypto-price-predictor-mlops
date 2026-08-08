# Start from an official lightweight Python image  this gives us Python
# pre-installed, so we don't have to set it up from scratch
FROM python:3.11-slim

# Set the working directory inside the container everything we do next
# happens relative to this folder
WORKDIR /app

# Copy just the requirements file first (not all your code yet)
COPY requirements.txt .

# Install the Python packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project files into the container
COPY . .

# Tell Docker this container will listen on port 8000 (same port uvicorn uses)
EXPOSE 8000

# The command that runs when the container starts 
# same command you've been running manually, just without --reload
# (--reload is a dev convenience, not needed/wanted in production)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]