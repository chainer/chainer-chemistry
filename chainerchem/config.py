import logging

# --- Configuration ---
DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


# --- Constant definitions ---
# The maximum atomic number in rdkit
MAX_ATOMIC_NUM = 117
