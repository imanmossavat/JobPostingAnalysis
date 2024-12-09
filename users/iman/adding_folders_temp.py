import os
import sys

# Adjust this string relative to the current file location to locate the root directory
string = "../../"  # Modify this path as needed
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), string))

# Print to verify the resolved root directory
print(f"Resolved root directory: {root_dir}")

# Ensure the resolved directory exists
if not os.path.isdir(root_dir):
    raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

# Check if the root directory ends with "JobPostingAnalysis"
expected_dir = "JobPostingAnalysis"
if not root_dir.endswith(expected_dir):
    raise ValueError(f"Root directory should end with '\\{expected_dir}' (or '/{expected_dir}' on non-Windows systems). Current path: {root_dir}")

# Add the resolved root directory to the Python path
if root_dir not in sys.path:
    sys.path.append(root_dir)
    print(f"Added {root_dir} to sys.path")
else:
    print(f"{root_dir} is already in sys.path")
