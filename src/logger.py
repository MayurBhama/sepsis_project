import logging
import os
import sys
from datetime import datetime
from logging import FileHandler, StreamHandler

# --- 1. SETUP LOG FILE PATH ---

# Define the root log directory
LOG_DIR = "sepsis_project/logs"
os.makedirs(LOG_DIR, exist_ok=True) # Create logs folder if it doesn't exist

# Create a unique log file name using the current date
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Define the universal format for ALL log messages
LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(module)s:%(lineno)d | %(message)s"
formatter = logging.Formatter(LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

# --- 2. CONFIGURE HANDLERS ---

# 2A. FILE HANDLER (Sends logs to the file)
file_handler = FileHandler(LOG_PATH)
file_handler.setLevel(logging.INFO) # Only log INFO and above to the file
file_handler.setFormatter(formatter)

# 2B. CONSOLE HANDLER (Prints logs to the terminal/console)
console_handler = StreamHandler(sys.stdout) # Use sys.stdout for console output
console_handler.setLevel(logging.INFO) # Only log INFO and above to the console
console_handler.setFormatter(formatter)

# --- 3. CONFIGURE LOGGER (The main entry point) ---

# Get the root logger (or use logging.getLogger(__name__) for a module logger)
logger = logging.getLogger() 
logger.setLevel(logging.INFO) # Global minimum level for all messages

# Prevent duplicate logs in case the root logger already has a default StreamHandler
logger.handlers.clear() 

# Attach the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
