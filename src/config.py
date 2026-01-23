import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch keys safely
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# Validation check
if not API_KEY or not API_SECRET:
    raise ValueError("❌ Missing Alpaca API keys! Please check your .env file.")