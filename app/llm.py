"""
LLM integration module for generating analysis code using Google's Gemini API.
Includes vision model support for image analysis.
"""
import os
import json
import asyncio
import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .logger import setup_logger, log_llm_interaction, logger
from .utils import analyze_file_structure, get_file_sample_content
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
MODEL_NAME = "gemini-1.5-flash"  # Standard model
VISION_MODEL_NAME = "gemini-1.5-flash"  # Vision-capable model
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Safety settings - allow code generation
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def detect_image_files(file_paths: List[Path]) -> List[Path]:
    """
    Detect image files from the list of file paths.
    
    Args:
        file_paths: List of file paths to check
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    return [fp for fp in file_paths if fp.suffix.lower() in image_extensions]

async def generate_vision_analysis_code(
    question: str,
    file_paths: List[Path],
    image_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path,
    request_id: str = "unknown"
) -> Optional[str]:
    """
    Generate Python analysis code using Gemini Vision API for image analysis.
    
    Args:
        question: Natural language analysis question
        file_paths: List of all file paths
        image_paths: List of image file paths
        analysis_type: Type of analysis requested
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        
    Returns:
        Generated Python code or None if generation fails
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    
    try:
        # Analyze non-image file structure
        non_image_files = [fp for fp in file_paths if fp not in image_paths]
        file_analysis = analyze_file_structure(non_image_files) if non_image_files else {}
        
        # Prepare images for vision model
        image_parts = []
        for image_path in image_paths:
            try:
                # Use PIL to load and prepare the image for Gemini
                from PIL import Image as PILImage
                img = PILImage.open(image_path)
                
                # Ensure image is in RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                image_parts.append(img)
                logger.info(f"Added image {image_path.name} to vision analysis")
            except Exception as e:
                logger.error(f"Error reading image {image_path}: {str(e)}")
        
        if not image_parts:
            logger.error("No valid images found for vision analysis")
            return None
        
        # Create vision prompt
        prompt = create_vision_analysis_prompt(
            question=question,
            file_analysis=file_analysis,
            file_paths=file_paths,
            image_paths=image_paths,
            analysis_type=analysis_type,
            sandbox_path=sandbox_path
        )
        
        # Initialize vision model
        model = genai.GenerativeModel(
            model_name=VISION_MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        
        # Prepare content for vision model (text + images)
        content_parts = [prompt] + image_parts
        
        # Generate response
        response = await asyncio.to_thread(model.generate_content, content_parts)
        
        if not response or not response.text:
            logger.error(f"Empty response from Gemini Vision API for request {request_id}")
            return None
        
        # Log interaction
        log_llm_interaction(
            logger=logger,
            request_id=request_id,
            prompt_length=len(prompt),
            response_length=len(response.text),
            model_used=f"{VISION_MODEL_NAME} (Vision)"
        )
        
        # Extract code from response
        generated_code = extract_code_from_response(response.text)
        
        if not generated_code:
            logger.error(f"Could not extract code from Gemini Vision response for request {request_id}")
            return None
        
        # Validate generated code
        from .utils import validate_generated_code
        is_valid, error_msg = validate_generated_code(generated_code)
        
        if not is_valid:
            logger.error(f"Generated vision code validation failed for request {request_id}: {error_msg}")
            return None
        
        logger.info(f"Successfully generated vision analysis code for request {request_id}")
        return generated_code
        
    except Exception as e:
        logger.error(f"Error generating vision analysis code for request {request_id}: {str(e)}")
        return None

async def generate_analysis_code(
    question: str,
    file_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path,
    request_id: str = "unknown"
) -> Optional[str]:
    """
    Generate Python analysis code using Gemini API.
    
    Args:
        question: Natural language analysis question
        file_paths: List of file paths to analyze
        analysis_type: Type of analysis requested
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        
    Returns:
        Generated Python code or None if generation fails
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    
    try:
        # Check for image files
        image_files = detect_image_files(file_paths)
        
        # Use vision model if images are present
        if image_files:
            logger.info(f"Detected {len(image_files)} image files, using vision model")
            return await generate_vision_analysis_code(
                question=question,
                file_paths=file_paths,
                image_paths=image_files,
                analysis_type=analysis_type,
                sandbox_path=sandbox_path,
                request_id=request_id
            )
        
        # Analyze file structure for text-only analysis
        file_analysis = analyze_file_structure(file_paths)
        
        # Create text-only prompt
        prompt = create_analysis_prompt(
            question=question,
            file_analysis=file_analysis,
            file_paths=file_paths,
            analysis_type=analysis_type,
            sandbox_path=sandbox_path
        )
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        
        # Generate response
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        if not response or not response.text:
            logger.error(f"Empty response from Gemini API for request {request_id}")
            return None
        
        # Log interaction
        prompt_length = len(prompt)
        log_llm_interaction(
            logger=logger,
            request_id=request_id,
            prompt_length=prompt_length,
            response_length=len(response.text),
            model_used=MODEL_NAME
        )
        
        # Extract code from response
        generated_code = extract_code_from_response(response.text)
        
        if not generated_code:
            logger.error(f"Could not extract code from Gemini response for request {request_id}")
            return None
        
        # Validate generated code
        from .utils import validate_generated_code
        is_valid, error_msg = validate_generated_code(generated_code)
        
        if not is_valid:
            logger.error(f"Generated code validation failed for request {request_id}: {error_msg}")
            return None
        
        logger.info(f"Successfully generated analysis code for request {request_id}")
        return generated_code
        
    except Exception as e:
        logger.error(f"Error generating analysis code for request {request_id}: {str(e)}")
        return None

def create_analysis_prompt(
    question: str,
    file_analysis: Dict[str, Any],
    file_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path
) -> str:
    """
    Create a detailed prompt for code generation.
    
    Args:
        question: User's analysis question
        file_analysis: Analysis of uploaded files
        file_paths: List of file paths
        analysis_type: Type of analysis
        sandbox_path: Path to sandbox directory
        
    Returns:
        Formatted prompt string
    """
    # Import here to avoid circular imports
    from .utils import get_all_available_files
    
    # Get all available files including any scraped data
    all_files = get_all_available_files(sandbox_path, file_paths)
    
    # Get sample content from files
    file_samples = {}
    for file_path in all_files[:5]:  # Limit to first 5 files for context
        sample = get_file_sample_content(file_path, max_lines=5)
        file_samples[file_path.name] = sample
    
    # Check if scraped data files exist
    scraped_files = [f for f in all_files if 'scraped_data' in f.name]
    scraped_context = ""
    if scraped_files:
        scraped_context = f"""

**PREVIOUSLY SCRAPED DATA AVAILABLE:**
The following scraped data files are available from previous analysis steps:
{json.dumps([f.name for f in scraped_files], indent=2)}
You can use these files directly in your analysis instead of scraping again.
"""
    
    prompt = f"""
You are a data analysis expert. Generate Python code to answer the user's question about the uploaded data files.

**USER QUESTION:** {question}

**ANALYSIS TYPE:** {analysis_type}

**AVAILABLE FILES:**
{json.dumps(file_analysis, indent=2)}

**FILE SAMPLES:**
{json.dumps(file_samples, indent=2)}{scraped_context}

**REQUIREMENTS:**
1. Write complete, executable Python code
2. Use only these allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, networkx, scipy, json, csv, base64, pathlib, requests, beautifulsoup4, duckdb, pyarrow, sqlparse, fsspec, s3fs, PIL
3. **CRITICAL: NO PLACEHOLDER CODE ALLOWED - For DuckDB S3 operations:** ALWAYS connect directly to real S3 buckets using DuckDB's httpfs extension. NEVER use placeholder text like "Replace with actual S3 path" or "metadata.parquet". Use the EXACT S3 URLs provided in the question.
4. **CRITICAL: USE ACTUAL DATA SOURCES** - If the question mentions S3 buckets, DuckDB queries, or specific data sources, use them EXACTLY as specified. NO placeholder files like "metadata.parquet" - use the real S3 paths.
5. **FOR IMAGE ANALYSIS:** Use PIL (Pillow) to load and analyze images: `from PIL import Image; img = Image.open('filename.jpg')`
6. Handle errors gracefully with try-except blocks
7. Include comments explaining the analysis approach
8. Save all results to specific output files:
   - JSON results: save to 'result.json' as a JSON array with exactly 4 elements: [numeric_answer, string_answer, float_answer, base64_image_string]

**SPECIFIC GUIDANCE BY ANALYSIS TYPE:**
{get_analysis_type_guidance(analysis_type)}

Generate the Python code now:
"""
    
    return prompt

def create_vision_analysis_prompt(
    question: str,
    file_analysis: Dict[str, Any],
    file_paths: List[Path],
    image_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path
) -> str:
    """
    Create a vision analysis prompt for image analysis.
    
    Args:
        question: User's analysis question
        file_analysis: Analysis of non-image files
        file_paths: List of all file paths
        image_paths: List of image file paths
        analysis_type: Type of analysis
        sandbox_path: Path to sandbox directory
        
    Returns:
        Formatted prompt string for vision model
    """
    # Import here to avoid circular imports
    from .utils import get_all_available_files
    
    # Get all available files including any scraped data
    all_files = get_all_available_files(sandbox_path, file_paths)
    
    # Get sample content from non-image files
    file_samples = {}
    for file_path in all_files[:5]:  # Limit to first 5 files for context
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            sample = get_file_sample_content(file_path, max_lines=5)
            file_samples[file_path.name] = sample
    
    prompt = f"""
You are a data analysis expert with vision capabilities. Generate Python code to answer the user's question using the provided files and images.

**USER QUESTION:** {question}

**ANALYSIS TYPE:** {analysis_type}

**NON-IMAGE FILES:**
{json.dumps(file_analysis, indent=2)}

**NON-IMAGE FILE SAMPLES:**
{json.dumps(file_samples, indent=2)}

**IMAGE FILES PROVIDED:** {', '.join([img.name for img in image_paths])}

**VISION CAPABILITIES:** You can see the content of the uploaded images. Use this visual information to extract data, read text, identify objects, or analyze visual elements as requested.

**REQUIREMENTS:**
1. Write complete, executable Python code
2. Use only these allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, networkx, scipy, json, csv, base64, pathlib, requests, beautifulsoup4, duckdb, pyarrow, sqlparse, fsspec, s3fs, PIL
3. **FOR IMAGE ANALYSIS:** Use PIL (Pillow) to load images and extract the visual information you can see
4. **VISION ANALYSIS:** You can see the image content - use this information to extract data, tables, text, or other relevant information
5. Handle errors gracefully with try-except blocks
6. Save results to 'result.json' as a JSON array with exactly 4 elements: [numeric_answer, string_answer, float_answer, base64_image_string]

**SPECIFIC GUIDANCE FOR IMAGE ANALYSIS:**
{get_analysis_type_guidance(analysis_type)}

Generate the Python code now:
"""
    
    return prompt

def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Extract Python code from the LLM response.
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        Extracted Python code or None if not found
    """
    # Look for code blocks
    import re
    
    # Try to find Python code blocks
    python_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(python_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks
    code_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        # Check if it looks like Python code
        code = matches[0].strip()
        if any(keyword in code for keyword in ['import ', 'def ', 'if __name__']):
            return code
    
    # If no code blocks found, try to extract code-like content
    lines = response_text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if any(keyword in line for keyword in ['import ', 'def ', 'try:', 'if ', 'for ', 'while ']):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None

def get_analysis_type_guidance(analysis_type: str) -> str:
    """Get specific guidance based on analysis type."""
    guidance = {
        "statistical": """
- Calculate descriptive statistics (mean, median, std, etc.)
- Perform correlation analysis if multiple numeric columns
- Create histograms and box plots
- Include statistical tests if appropriate
        """,
        
        "image": """
- **VISION ANALYSIS:** Use the visual information from uploaded images to extract data
- Use PIL (Pillow) for any image processing: `from PIL import Image; img = Image.open('filename.jpg')`
- Extract text, numbers, tables, or charts from images based on what you can see
- For data extraction: identify tables, charts, or text content in the image
- **CRITICAL:** The vision model can see image content - use this information directly
- Convert visual data to structured format (pandas DataFrame, lists, etc.)
- Create visualizations based on extracted data if requested
        """,
        
        "database": """
- **CRITICAL: Use REAL S3 connections, NO placeholders**
- Use DuckDB for efficient SQL-based analysis
- Connect directly to S3 buckets using httpfs extension - use EXACT URLs from the question
- Install and load required DuckDB extensions: httpfs, parquet, json
- Use proper S3 URLs with region specifications EXACTLY as provided
- Execute SQL queries directly on cloud data without downloading
- NEVER use placeholder files like "metadata.parquet" - use real S3 paths
- **PROPER DATE HANDLING:** Use STRPTIME() for VARCHAR dates, DATE_DIFF() for date arithmetic
        """,
        
        "general": """
- Explore data structure and quality first
- Create appropriate visualizations for the data type
- Provide meaningful insights and summaries
- **IMPORTANT: For S3 data sources or DuckDB queries, use actual S3 URLs - NO placeholders**
- Use DuckDB for large datasets or S3 data sources with real bucket paths
- Handle missing values and outliers appropriately
- **For image files:** Use vision capabilities to extract and analyze visual content
        """
    }
    
    return guidance.get(analysis_type, guidance["general"])

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the configured model.
    
    Returns:
        Dictionary with model information
    """
    return {
        "model_name": MODEL_NAME,
        "vision_model_name": VISION_MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY),
        "generation_config": GENERATION_CONFIG
    }
