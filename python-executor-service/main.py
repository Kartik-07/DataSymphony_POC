# MY_RAG/python-executor-service/main.py
import os
import subprocess
import sys
import traceback
import tempfile
import logging
import base64
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
# Importing uvicorn isn't strictly necessary for the code to run when started by uvicorn command

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure a directory for potential plot outputs within the container
ARTIFACT_DIR = tempfile.mkdtemp()
logger.info(f"Artifact directory created at: {ARTIFACT_DIR}")

# --- Pydantic Model for Request Body ---
class CodePayload(BaseModel):
    code: str

# --- FastAPI Endpoint ---
@app.post("/execute")
async def execute_code(payload: CodePayload):
    """
    Receives Python code via POST request body and executes it.
    """
    code = payload.code
    logger.info("Received code for execution.")

    # Very basic security check (can be expanded)
    if "import os" in code or "import subprocess" in code or "import sys" in code:
         logger.warning("Execution attempt with potentially unsafe imports blocked.")
         raise HTTPException(
             status_code=status.HTTP_403_FORBIDDEN,
             detail="Code contains potentially unsafe imports (os, subprocess, sys)."
         )

    stdout_result = ""
    stderr_result = ""
    plot_artifact_base64 = None
    script_path = None 

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_code_file:
            script_path = tmp_code_file.name
            # Prepend necessary imports to the user's code
            tmp_code_file.write("import pandas as pd\n")
            tmp_code_file.write("import numpy as np\n")
            tmp_code_file.write("import matplotlib\n")
            tmp_code_file.write("matplotlib.use('Agg') # Use non-interactive backend\n")
            tmp_code_file.write("import matplotlib.pyplot as plt\n")
            tmp_code_file.write("import seaborn as sns # Added Seaborn import\n") # MODIFIED LINE
            tmp_code_file.write("import scipy\n")
            # Ensure plots are saved to the correct directory within the container
            # Use a more robust way to set working directory for plotting if needed,
            # but saving to ARTIFACT_DIR with an absolute path is safer.
            # tmp_code_file.write(f"import os\nos.chdir('{ARTIFACT_DIR}')\n") # This line might be problematic if code relies on other relative paths
            tmp_code_file.write("\n# --- User Code Start ---\n")
            tmp_code_file.write(code)
            tmp_code_file.write("\n# --- User Code End ---\n")
            tmp_code_file.flush()

        logger.info(f"Executing code from temporary file: {script_path}")

        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,          
            text=True,                    
            timeout=60,                   
            check=False,
            cwd=ARTIFACT_DIR # Execute the script with ARTIFACT_DIR as the current working directory
        )

        stdout_result = process.stdout
        stderr_result = process.stderr

        logger.info(f"Execution finished. Return code: {process.returncode}")
        if stdout_result: logger.debug(f"STDOUT:\n{stdout_result}")
        if stderr_result: logger.warning(f"STDERR:\n{stderr_result}")

        plot_path = os.path.join(ARTIFACT_DIR, 'plot.png') # Matplotlib/Seaborn save here
        if os.path.exists(plot_path):
             logger.info("Plot artifact 'plot.png' found. Encoding...")
             with open(plot_path, 'rb') as f:
                 plot_artifact_base64 = base64.b64encode(f.read()).decode('utf-8')
             try:
                 os.remove(plot_path) 
             except OSError as e:
                 logger.error(f"Error deleting plot file {plot_path}: {e}")
        else:
            logger.info("No plot artifact 'plot.png' found in artifact directory.")


        return {
            "stdout": stdout_result,
            "stderr": stderr_result,
            "plot_png_base64": plot_artifact_base64,
            "execution_successful": process.returncode == 0
        }

    except subprocess.TimeoutExpired:
        logger.error("Code execution timed out.")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={
                "message": "Code execution timed out after 60 seconds.",
                "stdout": stdout_result,
                "stderr": stderr_result, # stderr_result might not be populated if timeout is aggressive
            }
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during code execution: {e}\n{traceback.format_exc()}")
        raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail={
                "message": f"Internal error during code execution: {e}",
                "stdout": stdout_result, # Include any partial output
                "stderr": stderr_result, # Include any partial error output
             }
         )
    finally:
        if script_path and os.path.exists(script_path):
            try:
                os.remove(script_path)
                logger.debug(f"Deleted temporary script: {script_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary script {script_path}: {e}")