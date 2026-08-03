#!/usr/bin/env bash

# Citation: Kędziorski, ‘Automate Your Python Environment’.

# Usage:
#   source setup_venv.sh [--help|--deactivate]

# Exit immediately if a command exits with a non-zero status
set -e

# Define the virtual environment directory
VENV_DIR=${VENV_DIR:-"venv"}

# Function to find the correct Python command
find_python() {
    if command -v python &> /dev/null; then
        echo "python"  # Fallback to python (Windows or systems with only python)
    elif command -v python3 &> /dev/null; then
        echo "python3"  # Prefer python3 on Linux/MacOS
    else
        echo "Error: Python is not installed or not in PATH." >&2
        return 1
    fi
}

# # Check if venv module is available
# check_venv_module() {
#     if ! "$PYTHON_CMD" -m venv --help &> /dev/null; then
#         echo "Error: The 'venv' module is not available. Please ensure Python is installed correctly." >&2
#         return 1  # Use 'return' instead of 'exit' to avoid closing the shell
#     fi
# }

# Function to activate the virtual environment
activate_venv() {
    if [ -f "$VENV_DIR/Scripts/activate" ]; then
        source "$VENV_DIR/Scripts/activate"  # Windows (Git Bash/WSL)
    elif [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"  # Linux/MacOS
    else
        echo "Error: Unable to find the activation script in '$VENV_DIR'." >&2
        return 1
    fi
}

# Help message
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: source setup_venv.sh"
    echo "Creates and activates a Python virtual environment in the 'venv' directory."
    echo "Options:"
    echo "  --help, -h       Show this help message."
    echo "  --deactivate, -d Deactivate the virtual environment."
    return 0  # Use 'return' instead of 'exit' to avoid closing the shell
fi

# Deactivate option
if [[ "$1" == "--deactivate" || "$1" == "-d" ]]; then
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
        echo "Virtual environment deactivated."
    else
        echo "No virtual environment is currently active."
    fi
    return 0  # Use 'return' instead of 'exit' to avoid closing the shell
fi

# Find the correct Python command
PYTHON_CMD=$(find_python)
if [ $? -ne 0 ]; then
    echo "$PYTHON_CMD"  # Print the error message
    return 1  # Use 'return' instead of 'exit' to avoid closing the shell
fi

# Check if venv module is available
# check_venv_module

# Check if a virtual environment is already active
if [ -n "$VIRTUAL_ENV" ]; then
    echo "A virtual environment is already activated: $VIRTUAL_ENV"
    return 0  # Use 'return' to prevent further execution
fi

# Check if the virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Activating it..."
    activate_venv
else
    echo "Virtual environment '$VENV_DIR' does not exist. Creating it..."

    # Create the virtual environment
    if ! "$PYTHON_CMD" -m venv "$VENV_DIR"; then
        echo "Error: Failed to create the virtual environment '$VENV_DIR'." >&2
        echo "Ensure you have the necessary permissions and Python is installed correctly." >&2
        return 1  # Use 'return' instead of 'exit' to avoid closing the shell
    fi

    echo "Virtual environment '$VENV_DIR' created successfully. Activating it..."
    activate_venv
fi

# Verify that the virtual environment is activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment '$VENV_DIR' is now active."
    return 0
else
    echo "Error: Virtual environment was not activated." >&2
    return 1  # Use 'return' instead of 'exit' to avoid closing the shell
fi

$SHELL