import subprocess

# Run main.py
subprocess.run(["python", "main.py"])

# Run add_missing_data.py
subprocess.run(["python", "interpolate.py"])

# Run visualize.py
subprocess.run(["python", "visualize.py"])