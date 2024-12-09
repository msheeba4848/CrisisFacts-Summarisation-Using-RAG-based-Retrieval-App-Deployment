# Frontend Documentation

## 1. Overview

This document provides a step-by-step guide on how to build, run, and test the **frontend** portion of the project using Docker. The frontend is a React application designed to serve as the user interface for the summarization system.

---

## 2. Directory Structure

The directory structure of the frontend project is as follows:

```plaintext
frontend/
├── Dockerfile.frontend    # Dockerfile for building and running the frontend app
├── package-lock.json      # Lock file for npm dependencies
├── package.json           # Project configuration and scripts
├── public/
│   └── index.html         # Main HTML file for React app
└── src/
    ├── App.js             # Main React component
    ├── index.js           # React entry point
    └── styles.css         # CSS styles for the application
```

---

## 3. Dockerfile

The `Dockerfile.frontend` defines the steps to build and run the frontend application inside a Docker container.

**Content of `Dockerfile.frontend`:**

```dockerfile
# Use the official Node.js image
FROM node:20

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy all source code into the container
COPY . .

# Expose the port for the React app
EXPOSE 3000

# Start the development server
CMD ["npm", "start"]
```

---

## 4. Building and Running the Frontend with Docker

Follow the steps below to build and run the frontend application using Docker.

### Step 1: Build the Docker Image

Navigate to the `frontend` directory and build the Docker image using the following command:

```bash
docker build -t frontend-image -f Dockerfile.frontend .
```

- **`-t frontend-image`**: Assigns a name to the Docker image.
- **`-f Dockerfile.frontend`**: Specifies the Dockerfile to use.

### Step 2: Run the Docker Container

Start a container using the image you just built:

```bash
docker run -d -p 3000:3000 --name frontend-container frontend-image
```

- **`-d`**: Runs the container in detached mode (background).
- **`-p 3000:3000`**: Maps port `3000` on the host to port `3000` in the container.
- **`--name frontend-container`**: Assigns a name to the container.

### Step 3: Access the Application

Once the container is running, open a web browser and go to:

```plaintext
http://localhost:3000
```

You should see the frontend application interface.

---

## 5. Verifying the Container

### Check Running Containers

To confirm that the container is running, use the following command:

```bash
docker ps
```

You should see `frontend-container` listed as one of the running containers.

### View Logs

To check the logs for the container, use:

```bash
docker logs frontend-container
```

---

## 6. Common Commands

### Stop the Running Container

To stop the frontend container, run:

```bash
docker stop frontend-container
```

### Restart the Container

To restart the stopped container, use:

```bash
docker start frontend-container
```

### Remove the Container

To remove the container, first stop it and then run:

```bash
docker rm frontend-container
```

### Rebuild the Image

If you make changes to the source code or Dockerfile, rebuild the image with:

```bash
docker build --no-cache -t frontend-image -f Dockerfile.frontend .
```

---

## 7. Notes

1. **Port Mapping**: The React development server runs on port `3000` by default. Ensure the port is not being used by any other process.

2. **`.dockerignore`**: To avoid copying unnecessary files (e.g., `node_modules`), make sure to include a `.dockerignore` file with the following content:

   ```plaintext
   node_modules
   build
   .git
   Dockerfile*
   npm-debug.log
   ```

3. **Code Changes**: In development mode (`npm start`), any changes to the source code are **not** reflected automatically in the container. To apply changes, rebuild the image and restart the container.

---
