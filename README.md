**CSCE 482 Capstone Project**

## Running with Docker

To run the Jupyter notebook environment using Docker:

1. Build the Docker image:
   ```bash
   docker build -t jupyter-notebook .
   ```

2. Run the container:

   **Mac/Linux:**
   ```bash
   docker run -p 8888:8888 -v $(pwd):/app jupyter-notebook
   ```

   **Windows (Command Prompt):**
   ```cmd
   docker run -p 8888:8888 -v %cd%:/app jupyter-notebook
   ```

   **Windows (PowerShell):**
   ```powershell
   docker run -p 8888:8888 -v ${PWD}:/app jupyter-notebook
   ```

3. Open your browser and navigate to `http://localhost:8888`

The notebook will be accessible without a password for development purposes.
