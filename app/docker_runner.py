"""
Docker container runner for executing LLM-generated Python code safely.
Optimized for efficiency with pre-built images and smart package management.
"""
import os
import time
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil

from .logger import setup_logger, log_docker_operation, logger
from .utils import create_code_file, read_execution_results

# Initialize logger (use global logger from logger module)
# logger = setup_logger()

# Docker configuration - use only optimized analysis image
ANALYSIS_IMAGE = "data-analysis-env:latest"
DOCKER_TIMEOUT = int(os.getenv("DOCKER_TIMEOUT", 300))  # 5 minutes max execution time

# Cache for checking image availability
_image_cache = {}

async def initialize_docker_environment() -> Dict[str, Any]:
    """
    Initialize Docker environment for optimal performance.
    This should be called during application startup.
    
    Returns:
        Dictionary with initialization status and details
    """
    logger.info("Initializing Docker environment for data analysis...")
    
    # Check Docker availability
    docker_available = await check_docker_availability()
    if not docker_available:
        logger.error("Docker is not available - analysis will fail")
        return {
            "docker_available": False,
            "optimized_image_available": False,
            "fallback_image_available": False,
            "status": "error",
            "message": "Docker is not available"
        }
    
    # Ensure optimized analysis image is available
    optimized_available = await ensure_analysis_image_available()
    
    # Check fallback image availability
    fallback_available = await pull_docker_image()
    
    # Get Docker stats for monitoring
    docker_stats = await get_docker_stats()
    
    status = "ready" if (optimized_available or fallback_available) else "degraded"
    
    result = {
        "docker_available": docker_available,
        "optimized_image_available": optimized_available,
        "fallback_image_available": fallback_available,
        "docker_stats": docker_stats,
        "status": status,
        "message": f"Docker environment initialized - using {'optimized' if optimized_available else 'fallback'} execution mode"
    }
    
    logger.info(f"Docker initialization complete: {result['message']}")
    return result

async def execute_code_in_docker(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute Python code safely in a Docker container with optimized efficiency.
    
    Optimizations:
    - Use pre-built analysis image when available
    - Smart package installation only when needed
    - Efficient container startup and cleanup
    - Network isolation for security
    
    Args:
        code: Python code to execute
        sandbox_path: Path to sandbox directory with files
        request_id: Request ID for logging
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    start_time = time.time()
    
    try:
        # Create code file
        code_file = create_code_file(code, sandbox_path)
        logger.info(f"Request {request_id}: Created code file: {code_file}")
        
        # Determine best image to use
        image_name = await get_best_available_image()
        logger.info(f"Request {request_id}: Using Docker image: {image_name}")
        
        # Execute with the optimal strategy based on image type
        if image_name == ANALYSIS_IMAGE:
            # Use optimized execution for pre-built image
            return await execute_with_prebuilt_image(
                code=code,
                sandbox_path=sandbox_path,
                request_id=request_id,
                timeout=timeout,
                image_name=image_name
            )
        else:
            # Use fallback execution with package installation
            return await execute_with_package_install(
                code=code,
                sandbox_path=sandbox_path,
                request_id=request_id,
                timeout=timeout,
                image_name=image_name
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Request {request_id}: Docker execution error: {str(e)}")
        
        return {
            "success": False,
            "error": f"Docker execution failed: {str(e)}",
            "execution_time": execution_time,
            "stdout": "",
            "stderr": str(e),
            "results": {}
        }

async def get_best_available_image() -> str:
    """
    Determine if the analysis image is available.
    
    Returns:
        Name of the analysis Docker image if available, raises exception if not
    """
    global _image_cache
    
    # Check cache first
    if ANALYSIS_IMAGE in _image_cache:
        if _image_cache[ANALYSIS_IMAGE]:
            return ANALYSIS_IMAGE
        else:
            raise RuntimeError("Analysis image not available and no fallback configured")
    
    # Check if analysis image exists
    try:
        check_cmd = ["docker", "images", "-q", ANALYSIS_IMAGE]
        process = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        
        image_exists = bool(stdout_data.decode().strip())
        _image_cache[ANALYSIS_IMAGE] = image_exists
        
        if image_exists:
            logger.info(f"Using analysis image: {ANALYSIS_IMAGE}")
            return ANALYSIS_IMAGE
        else:
            logger.error(f"Analysis image {ANALYSIS_IMAGE} not found and no fallback configured")
            raise RuntimeError(f"Analysis image {ANALYSIS_IMAGE} not available")
            
    except Exception as e:
        logger.error(f"Error checking for analysis image: {e}")
        _image_cache[ANALYSIS_IMAGE] = False
        raise RuntimeError(f"Failed to check analysis image availability: {e}")

async def execute_with_prebuilt_image(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int,
    image_name: str
) -> Dict[str, Any]:
    """
    Execute code using pre-built image with all packages already installed.
    This is the most efficient execution path.
    """
    start_time = time.time()
    
    try:
        log_docker_operation(logger, request_id, "start_optimized")
        
        # Check if code requires network access (web scraping)
        needs_network = check_if_network_needed(code)
        network_config = [] if needs_network else ["--network", "none"]
        
        if needs_network:
            logger.info(f"Request {request_id}: Enabling network access for web scraping")
        else:
            logger.info(f"Request {request_id}: Using isolated network for security")
        
        # Prepare optimized Docker command for pre-built image
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            *network_config,  # Conditional network access
            "--memory", "768m",  # Slightly more memory for pre-built image
            "--cpus", "1.5",  # Allow more CPU for faster execution
            "--name", f"analysis-opt-{request_id[:8]}",
            "-v", f"{sandbox_path.absolute()}:/workspace",
            "-w", "/workspace",
            "--user", "1000:1000",  # Run as non-root user
            image_name,
            "python", "analysis.py"
        ]
        
        # Execute Docker container
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for completion with timeout
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            await cleanup_timed_out_container(f"analysis-opt-{request_id[:8]}", process)
            
            execution_time = time.time() - start_time
            log_docker_operation(logger, request_id, "timeout", duration=execution_time)
            
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": execution_time,
                "stdout": "",
                "stderr": f"Process timed out after {timeout} seconds",
                "results": read_execution_results(sandbox_path)
            }
        
        execution_time = time.time() - start_time
        stdout_str = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
        stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
        
        success = process.returncode == 0
        results = read_execution_results(sandbox_path)
        
        log_docker_operation(logger, request_id, "complete_optimized", duration=execution_time)
        
        # Monitor performance
        await monitor_execution_performance(request_id, {
            "execution_time": execution_time,
            "success": success
        }, image_name)
        
        return {
            "success": success,
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "results": results,
            "error": None if success else f"Process exited with code {process.returncode}"
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Request {request_id}: Optimized execution failed: {str(e)}")
        
        return {
            "success": False,
            "error": f"Optimized Docker execution failed: {str(e)}",
            "execution_time": execution_time,
            "stdout": "",
            "stderr": str(e),
            "results": {}
        }

async def execute_with_package_install(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int,
    image_name: str
) -> Dict[str, Any]:
    """
    Execute code with package installation for fallback image.
    Uses smart package installation with caching.
    """
    start_time = time.time()
    
    try:
        log_docker_operation(logger, request_id, "start_fallback")
        
        # Check if code requires network access (web scraping)
        needs_network = check_if_network_needed(code)
        network_config = [] if needs_network else ["--network", "none"]
        
        if needs_network:
            logger.info(f"Request {request_id}: Enabling network access for web scraping (fallback mode)")
        else:
            logger.info(f"Request {request_id}: Using isolated network for security (fallback mode)")
        
        # Create optimized install script with better error handling
        install_script = """#!/bin/bash
set -e
echo "Installing required packages..."
pip install --no-cache-dir --quiet --disable-pip-version-check \\
    pandas==2.1.4 numpy==1.24.4 matplotlib==3.7.4 seaborn==0.13.0 \\
    networkx==3.2.1 scipy==1.11.4 plotly==5.17.0 \\
    beautifulsoup4==4.12.2 requests==2.31.0 duckdb==0.9.2 \\
    openpyxl==3.1.2 sqlparse==0.4.4 pyarrow==14.0.1 \\
    fsspec==2023.12.2 s3fs==2023.12.2 lxml==4.9.3 xlrd==2.0.1 \\
    ipython==8.18.1 psutil==5.9.6 python-dateutil==2.8.2 pytz==2023.3 2>/dev/null || {
    echo "Package installation failed, proceeding anyway..."
}
echo "Starting analysis..."
cd /workspace
python analysis.py
"""
        
        # Create the script file
        script_file = sandbox_path / "run_analysis.sh"
        with open(script_file, 'w') as f:
            f.write(install_script)
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        # Prepare Docker command for fallback execution
        docker_cmd = [
            "docker", "run",
            "--rm",
            *network_config,  # Conditional network access
            "--memory", "512m",
            "--cpus", "1.0",
            "--name", f"analysis-fb-{request_id[:8]}",
            "-v", f"{sandbox_path.absolute()}:/workspace",
            "-w", "/workspace",
            image_name,
            "bash", "run_analysis.sh"
        ]
        
        # Execute Docker container
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for completion with timeout
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            await cleanup_timed_out_container(f"analysis-fb-{request_id[:8]}", process)
            
            execution_time = time.time() - start_time
            log_docker_operation(logger, request_id, "timeout", duration=execution_time)
            
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": execution_time,
                "stdout": "",
                "stderr": f"Process timed out after {timeout} seconds",
                "results": read_execution_results(sandbox_path)
            }
        
        execution_time = time.time() - start_time
        stdout_str = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
        stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
        
        success = process.returncode == 0
        results = read_execution_results(sandbox_path)
        
        log_docker_operation(logger, request_id, "complete_fallback", duration=execution_time)
        
        # Monitor performance
        await monitor_execution_performance(request_id, {
            "execution_time": execution_time,
            "success": success
        }, image_name)
        
        return {
            "success": success,
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "results": results,
            "error": None if success else f"Process exited with code {process.returncode}"
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Request {request_id}: Fallback execution failed: {str(e)}")
        
        return {
            "success": False,
            "error": f"Fallback Docker execution failed: {str(e)}",
            "execution_time": execution_time,
            "stdout": "",
            "stderr": str(e),
            "results": {}
        }

async def cleanup_timed_out_container(container_name: str, process: asyncio.subprocess.Process) -> None:
    """
    Clean up a timed-out container and process.
    """
    try:
        # Kill the subprocess first
        process.kill()
        await process.wait()
        
        # Try to stop and remove the Docker container
        await asyncio.create_subprocess_exec(
            "docker", "stop", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.sleep(1)  # Give it a moment to stop
        await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
    except Exception as cleanup_error:
        logger.warning(f"Error cleaning up container {container_name}: {cleanup_error}")

async def check_docker_availability() -> bool:
    """
    Check if Docker is available and running.
    
    Returns:
        True if Docker is available, False otherwise
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.wait_for(process.communicate(), timeout=10)
        return process.returncode == 0
        
    except Exception:
        return False

async def pull_docker_image() -> bool:
    """
    Ensure the analysis Docker image is available, build it if necessary.
    
    Returns:
        True if image is available/built successfully, False otherwise
    """
    try:
        # Check if image exists locally
        check_cmd = ["docker", "images", "-q", ANALYSIS_IMAGE]
        process = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        
        if stdout_data.decode().strip():
            logger.info(f"Analysis image {ANALYSIS_IMAGE} already available locally")
            return True
        
        # Build the image if it doesn't exist
        logger.info(f"Analysis image {ANALYSIS_IMAGE} not found, building it...")
        return await build_analysis_image()
        
    except Exception as e:
        logger.error(f"Error checking/building analysis image: {str(e)}")
        return False

async def ensure_analysis_image_available() -> bool:
    """
    Ensure the analysis image is available, build it if necessary.
    
    Returns:
        True if image is available, False otherwise
    """
    try:
        # Check if analysis image exists
        check_cmd = ["docker", "images", "-q", ANALYSIS_IMAGE]
        process = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        
        if stdout_data.decode().strip():
            logger.info(f"Analysis image {ANALYSIS_IMAGE} already available")
            return True
        
        # Image doesn't exist, try to build it
        logger.info(f"Building analysis image: {ANALYSIS_IMAGE}")
        return await build_analysis_image()
        
    except Exception as e:
        logger.error(f"Error checking analysis image availability: {e}")
        return False

async def build_analysis_image() -> bool:
    """
    Build the optimized analysis Docker image using the analysis Dockerfile.
    
    Returns:
        True if build successful, False otherwise
    """
    try:
        # Check if Dockerfile.analysis exists
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile.analysis"
        requirements_path = Path(__file__).parent.parent / "analysis-requirements.txt"
        
        if not dockerfile_path.exists():
            logger.warning("Dockerfile.analysis not found, creating optimized dockerfile")
            return await create_and_build_analysis_image()
        
        if not requirements_path.exists():
            logger.warning("analysis-requirements.txt not found, creating default requirements")
            await create_analysis_requirements()
        
        # Build the image
        build_cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", ANALYSIS_IMAGE,
            str(dockerfile_path.parent)
        ]
        
        logger.info("Starting Docker image build (this may take a few minutes)...")
        process = await asyncio.create_subprocess_exec(
            *build_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, stderr_data = await asyncio.wait_for(
            process.communicate(),
            timeout=600  # 10 minutes timeout for build
        )
        
        if process.returncode == 0:
            logger.info(f"Successfully built analysis image: {ANALYSIS_IMAGE}")
            # Update cache
            _image_cache[ANALYSIS_IMAGE] = True
            return True
        else:
            stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
            logger.error(f"Failed to build analysis image: {stderr_str}")
            return False
            
    except asyncio.TimeoutError:
        logger.error("Docker image build timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Error building analysis image: {str(e)}")
        return False

async def create_and_build_analysis_image() -> bool:
    """
    Create a Dockerfile and build the analysis image from scratch.
    """
    try:
        # Create optimized Dockerfile content
        dockerfile_content = """FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for data science packages
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    libfreetype6-dev \\
    libpng-dev \\
    libjpeg-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir \\
    pandas==2.1.4 \\
    numpy==1.24.4 \\
    matplotlib==3.7.4 \\
    seaborn==0.13.0 \\
    networkx==3.2.1 \\
    scipy==1.11.4 \\
    plotly==5.17.0 \\
    beautifulsoup4==4.12.2 \\
    requests==2.31.0 \\
    duckdb==0.9.2 \\
    openpyxl==3.1.2 \\
    sqlparse==0.4.4 \\
    pyarrow==14.0.1 \\
    fsspec==2023.12.2 \\
    s3fs==2023.12.2 \\
    lxml==4.9.3 \\
    xlrd==2.0.1 \\
    ipython==8.18.1 \\
    psutil==5.9.6 \\
    python-dateutil==2.8.2 \\
    pytz==2023.3

# Pre-install DuckDB extensions for better performance
RUN python -c "import duckdb; conn = duckdb.connect(); conn.execute('INSTALL httpfs'); conn.execute('INSTALL parquet'); conn.execute('INSTALL json'); conn.execute('INSTALL fts'); conn.close()" || echo "Extension installation will be handled at runtime"

# Create non-root user for security
RUN useradd -m -u 1000 analysis && \\
    chown -R analysis:analysis /workspace

USER analysis

# Default command
CMD ["python"]
"""
        
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            
            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build image
            build_cmd = [
                "docker", "build",
                "-t", ANALYSIS_IMAGE,
                temp_dir
            ]
            
            logger.info("Building analysis image from generated Dockerfile...")
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=600  # 10 minutes timeout
            )
            
            if process.returncode == 0:
                logger.info(f"Successfully built analysis image: {ANALYSIS_IMAGE}")
                _image_cache[ANALYSIS_IMAGE] = True
                return True
            else:
                stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
                logger.error(f"Failed to build optimized image: {stderr_str}")
                return False
                
    except Exception as e:
        logger.error(f"Error creating and building analysis image: {str(e)}")
        return False

async def create_analysis_requirements() -> None:
    """
    Create analysis-requirements.txt file with optimized package versions.
    """
    requirements_content = """# Core data processing packages
pandas==2.1.4
numpy==1.24.4

# Visualization packages
matplotlib==3.7.4
seaborn==0.13.0
plotly==5.17.0

# Network analysis
networkx==3.2.1

# Scientific computing
scipy==1.11.4

# Database and SQL processing
duckdb==0.9.2
sqlparse==0.4.4
pyarrow==14.0.1

# Additional DuckDB-related packages for S3 and cloud storage
fsspec==2023.12.2
s3fs==2023.12.2

# File processing
openpyxl==3.1.2
xlrd==2.0.1
beautifulsoup4==4.12.2
lxml==4.9.3

# HTTP requests
requests==2.31.0

# Jupyter support (for notebook-style outputs)
ipython==8.18.1

# Additional data processing utilities
psutil==5.9.6
python-dateutil==2.8.2
pytz==2023.3
"""
    
    requirements_path = Path(__file__).parent.parent / "analysis-requirements.txt"
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    
def create_dockerfile_content() -> str:
    """
    Create Dockerfile content for the analysis environment.
    
    Returns:
        Dockerfile content as string
    """
    return """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    libfreetype6-dev \\
    libpng-dev \\
    libjpeg-dev \\
    pkg-config \\
    sqlite3 \\
    libsqlite3-dev \\
    curl \\
    unzip \\
    ca-certificates \\
    libssl-dev \\
    libcurl4-openssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    pandas==2.1.4 \\
    numpy==1.24.4 \\
    matplotlib==3.7.4 \\
    seaborn==0.13.0 \\
    networkx==3.2.1 \\
    scipy==1.11.4 \\
    plotly==5.17.0 \\
    beautifulsoup4==4.12.2 \\
    requests==2.31.0 \\
    duckdb==0.9.2 \\
    openpyxl==3.1.2 \\
    sqlparse==0.4.4 \\
    pyarrow==14.0.1 \\
    fsspec==2023.12.2 \\
    s3fs==2023.12.2 \\
    lxml==4.9.3 \\
    xlrd==2.0.1 \\
    ipython==8.18.1 \\
    psutil==5.9.6 \\
    python-dateutil==2.8.2 \\
    pytz==2023.3

# Pre-install DuckDB extensions for better performance
RUN python -c "import duckdb; conn = duckdb.connect(); conn.execute('INSTALL httpfs'); conn.execute('INSTALL parquet'); conn.execute('INSTALL json'); conn.execute('INSTALL fts'); conn.close()" || echo "Extension installation will be handled at runtime"

# Create non-root user
RUN useradd -m -u 1000 analyst
USER analyst

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python"]
"""

def build_custom_docker_image() -> bool:
    """
    Build custom Docker image with required packages.
    
    Returns:
        True if build successful, False otherwise
    """
    try:
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            
            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(create_dockerfile_content())
            
            # Build image
            build_cmd = [
                "docker", "build",
                "-t", "data-analysis-env:latest",
                temp_dir
            ]
            
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Successfully built custom Docker image")
                return True
            else:
                logger.error(f"Failed to build Docker image: {result.stderr}")
                return False
                
    except Exception as e:
        logger.error(f"Error building Docker image: {str(e)}")
        return False

async def execute_with_custom_image(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute code using custom Docker image with pre-installed packages.
    
    Args:
        code: Python code to execute
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    # Use analysis image - ensure it's available
    image_name = ANALYSIS_IMAGE
    
    # Check if analysis image exists
    try:
        check_cmd = ["docker", "images", "-q", image_name]
        process = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await process.communicate()
        
        if not stdout_data.decode().strip():
            # Analysis image doesn't exist, build it
            logger.error(f"Analysis image {image_name} not found")
            raise RuntimeError(f"Analysis image {image_name} not available")
            
    except Exception as e:
        logger.error(f"Failed to verify analysis image: {e}")
        raise RuntimeError(f"Analysis image verification failed: {e}")
    
    # Execute with the selected image
    return await execute_code_with_image(
        code=code,
        sandbox_path=sandbox_path,
        request_id=request_id,
        timeout=timeout,
        image_name=image_name
    )

async def execute_code_with_image(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int,
    image_name: str
) -> Dict[str, Any]:
    """
    Execute code with a specific Docker image.
    
    Args:
        code: Python code to execute
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        timeout: Maximum execution time in seconds
        image_name: Docker image to use
        
    Returns:
        Dictionary with execution results
    """
    start_time = time.time()
    
    try:
        # Create code file
        code_file = create_code_file(code, sandbox_path)
        
        # Install packages script for standard Python image
        install_script = """
pip install --no-cache-dir pandas==2.1.4 numpy==1.24.4 matplotlib==3.7.4 seaborn==0.13.0 networkx==3.2.1 scipy==1.11.4 plotly==5.17.0 beautifulsoup4==4.12.2 requests==2.31.0 duckdb==0.9.2 openpyxl==3.1.2 sqlparse==0.4.4 pyarrow==14.0.1 fsspec==2023.12.2 s3fs==2023.12.2 lxml==4.9.3 xlrd==2.0.1 ipython==8.18.1 psutil==5.9.6 python-dateutil==2.8.2 pytz==2023.3 2>/dev/null || echo "Package installation failed" && python analysis.py
"""
        
        # Create install script file
        install_file = sandbox_path / "install_and_run.sh"
        with open(install_file, 'w') as f:
            f.write(install_script)
        
        # Make script executable
        os.chmod(install_file, 0o755)
        
        # Prepare Docker command
        needs_network = check_if_network_needed(code)
        network_config = [] if needs_network else ["--network", "none"]
        
        if image_name != ANALYSIS_IMAGE:
            # Use install script for standard Python image
            cmd_args = ["bash", "install_and_run.sh"]
        else:
            # Use direct Python execution for analysis image
            cmd_args = ["python", "analysis.py"]
        
        docker_cmd = [
            "docker", "run",
            "--rm",
            *network_config,  # Conditional network access
            "--memory", "512m",
            "--cpus", "1.0",
            "-v", f"{sandbox_path.absolute()}:/workspace",
            "-w", "/workspace",
            "--user", "1000:1000",
            image_name,
            *cmd_args
        ]
        
        # Execute Docker container
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except:
                pass
            
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": time.time() - start_time,
                "stdout": "",
                "stderr": f"Process timed out after {timeout} seconds",
                "results": {}
            }
        
        execution_time = time.time() - start_time
        stdout_str = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
        stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
        
        success = process.returncode == 0
        results = read_execution_results(sandbox_path)
        
        return {
            "success": success,
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "results": results,
            "error": None if success else f"Process exited with code {process.returncode}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution failed: {str(e)}",
            "execution_time": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
            "results": {}
        }

def cleanup_docker_resources():
    """Clean up any dangling Docker resources."""
    try:
        # Remove dangling images
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True,
            timeout=30
        )
        
        # Remove dangling containers
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            capture_output=True,
            timeout=30
        )
        
        logger.info("Docker cleanup completed")
        
    except Exception as e:
        logger.warning(f"Docker cleanup failed: {str(e)}")

async def get_docker_stats() -> Dict[str, Any]:
    """
    Get Docker system statistics.
    
    Returns:
        Dictionary with Docker stats
    """
    try:
        # Get Docker info
        process = await asyncio.create_subprocess_exec(
            "docker", "info", "--format", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await process.communicate()
        
        if process.returncode == 0:
            info = json.loads(stdout_data.decode())
            return {
                "available": True,
                "containers": info.get("Containers", 0),
                "images": info.get("Images", 0),
                "server_version": info.get("ServerVersion", "unknown"),
                "memory_limit": info.get("MemTotal", 0)
            }
        else:
            return {"available": False, "error": "Docker not accessible"}
            
    except Exception as e:
        return {"available": False, "error": str(e)}

class DockerExecutor:
    """Context manager for Docker execution with cleanup."""
    
    def __init__(self, sandbox_path: Path, request_id: str):
        self.sandbox_path = sandbox_path
        self.request_id = request_id
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        log_docker_operation(self.logger, self.request_id, "initialize")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            log_docker_operation(self.logger, self.request_id, "cleanup", duration=duration)
        else:
            logger.error(f"Docker execution failed for {self.request_id}: {exc_val}")
        
        # Cleanup any resources if needed
        cleanup_docker_resources()
        
    async def execute(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute code and return results."""
        return await execute_code_in_docker(
            code=code,
            sandbox_path=self.sandbox_path,
            request_id=self.request_id,
            timeout=timeout
        )

def run_code_in_docker(code_str: str, input_dir: str) -> dict:
    """
    Synchronous wrapper to run Python code in a Docker container.
    
    Args:
        code_str: Python code to execute
        input_dir: Directory path containing input files (will be used as sandbox)
        
    Returns:
        Dictionary with execution results containing:
        - success: Boolean indicating if execution was successful
        - stdout: Standard output from execution
        - stderr: Standard error from execution
        - execution_time: Time taken for execution in seconds
        - return_code: Process return code
        - error: Error message if execution failed
    """
    import subprocess
    import time
    
    try:
        # Convert input_dir to Path object
        input_path = Path(input_dir)
        
        # Create input directory if it doesn't exist
        input_path.mkdir(parents=True, exist_ok=True)
        
        # Save code to script.py in the input directory
        script_path = input_path / "script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code_str)
        
        # Check if network access is needed
        needs_network = check_if_network_needed(code_str)
        network_config = [] if needs_network else ["--network", "none"]
        
        # Prepare Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            *network_config,  # Conditional network access
            "--memory", "512m",  # Memory limit
            "-v", f"{input_path.absolute()}:/sandbox",  # Mount input_dir as /sandbox
            "python:3.11-slim",  # Base image
            "python", "/sandbox/script.py"  # Command to execute
        ]
        
        # Record start time
        start_time = time.time()
        
        # Execute Docker container using subprocess.run()
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Decode output
        stdout_str = result.stdout if result.stdout else ""
        stderr_str = result.stderr if result.stderr else ""
        
        # Check if execution was successful
        success = result.returncode == 0
        
        return {
            "success": success,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "error": None if success else f"Process exited with code {result.returncode}"
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Process timed out after 300 seconds",
            "execution_time": 300,
            "return_code": -1,
            "error": "Execution timed out"
        }
    except Exception as e:
        logger.error(f"Error in run_code_in_docker: {str(e)}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "execution_time": 0,
            "return_code": -1,
            "error": f"Failed to execute code: {str(e)}"
        }

async def monitor_execution_performance(
    request_id: str, 
    execution_result: Dict[str, Any], 
    image_used: str
) -> Dict[str, Any]:
    """
    Monitor and log execution performance metrics.
    
    Args:
        request_id: Request identifier
        execution_result: Results from code execution
        image_used: Docker image that was used
        
    Returns:
        Performance metrics dictionary
    """
    performance_metrics = {
        "request_id": request_id,
        "image_used": image_used,
        "execution_time": execution_result.get("execution_time", 0),
        "success": execution_result.get("success", False),
        "optimization_level": "optimized" if image_used == ANALYSIS_IMAGE else "fallback",
        "timestamp": time.time()
    }
    
    # Log performance for monitoring
    if performance_metrics["optimization_level"] == "optimized":
        logger.info(f"PERFORMANCE | {request_id} | Optimized execution: {performance_metrics['execution_time']:.2f}s")
    else:
        logger.info(f"PERFORMANCE | {request_id} | Fallback execution: {performance_metrics['execution_time']:.2f}s")
    
    # Suggest optimizations if execution was slow
    if performance_metrics["execution_time"] > 60:
        if performance_metrics["optimization_level"] == "fallback":
            logger.info(f"OPTIMIZATION | {request_id} | Consider building optimized image for better performance")
        else:
            logger.info(f"OPTIMIZATION | {request_id} | Long execution time may indicate complex analysis")
    
    return performance_metrics

def check_if_network_needed(code: str) -> bool:
    """
    Check if the generated code requires network access for web scraping.
    
    Args:
        code: Python code to analyze
        
    Returns:
        True if network access is needed, False otherwise
    """
    network_indicators = [
        # Direct HTTP libraries
        'requests.get',
        'requests.post', 
        'requests.put',
        'requests.delete',
        'requests.head',
        'requests.Session',
        'session.get',
        'session.post',
        'urllib.request',
        'urllib.urlopen',
        'httplib',
        'http.client',
        
        # Web scraping specific
        'BeautifulSoup',
        'soup = BeautifulSoup',
        'response = requests',
        'html.parser',
        'lxml',
        
        # URLs and protocols
        'http://',
        'https://',
        'ftp://',
        'www.',
        's3://',  # DuckDB S3 access
        
        # Specific sites mentioned in scraping tasks
        'wikipedia.org',
        'en.wikipedia',
        'api.',
        '.com',
        '.org',
        '.net',
        
        # DuckDB S3 operations
        'install httpfs',
        'load httpfs',
        'read_parquet',
        's3_region',
        'indian-high-court-judgments',  # Specific S3 bucket mentioned in examples
        
        # Common web scraping patterns
        'response.content',
        'response.text',
        'soup.find',
        'soup.select',
        'soup.get_text',
        'selenium',
        'webdriver',
        
        # Network imports
        'import requests',
        'from requests',
        'import urllib',
        'from urllib',
        'import http',
        'from http'
    ]
    
    code_lower = code.lower()
    for indicator in network_indicators:
        if indicator.lower() in code_lower:
            logger.debug(f"Network access needed - found indicator: {indicator}")
            return True
    
    return False
