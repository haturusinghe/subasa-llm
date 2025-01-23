import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_variables():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parents[2] / '.env'
    load_dotenv(env_path)

    required_vars = ['HF_TOKEN', 'WANDB_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file or environment variables."
        )
