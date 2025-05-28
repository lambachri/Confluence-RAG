# Configuration file for the OpCon chatbot

import os
import sys
from dotenv import load_dotenv, find_dotenv

# Load environment variables
sys.path.append('../..')
_ = load_dotenv(find_dotenv())

# Confluence API access
CONFLUENCE_SPACE_NAME = os.getenv("CONFLUENCE_SPACE_NAME", "https://your-confluence-instance.atlassian.net")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "OPCON")
CONFLUENCE_USERNAME = os.getenv("EMAIL_ADRESS", "your-username@example.com")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY", "your-api-key")

# Vector database
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "vectorstore/chroma_db")

# Evaluation dataset
EVALUATION_DATASET = os.getenv("EVALUATION_DATASET", "data/evaluation/eval_dataset.tsv")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# List of topics the chatbot can handle (for filtering)
OPCON_TOPICS = [
    "opcon",
    "scheduler",
    "job", 
    "monitoring",
    "batch processing",
    "automation",
    "scheduling",
    "workflow",
    "task management",
    "error handling",
    "log analysis",
    "alerts",
    "notifications",
    "dependencies",
    "triggers",
    "scripts",
    "execution",
    "status",
    "reporting"
]

# List of common off-topic keywords to help with filtering
OFFTOPIC_KEYWORDS = [
    "password",
    "login",
    "email",
    "account",
    "website",
    "mobile",
    "phone",
    "laptop",
    "printer",
    "wifi",
    "network",
    "anaplan",
    "dior",
    "christian",
    "okta",
    "vxl",
    "quadrigramme"
]
