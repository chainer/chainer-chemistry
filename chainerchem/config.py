import logging

# --- Configuration ---
DEBUG = True
if DEBUG:
    logging.basicConfig(level=logging.WARNING)


# --- Constant definitions ---
# The maximum atomic number in rdkit
MAX_ATOMIC_NUM = 117
