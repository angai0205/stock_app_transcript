#!/usr/bin/env python3
"""
Simple script to view transcriber logs in real-time from another terminal.
Run this in a separate terminal while the main transcriber is running.
"""

import os
import sys
import time

def view_log_file():
    """View the log file in real-time"""
    log_file = "transcriber.log"
    
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' not found. Make sure the transcriber is running.")
        return
    
    print(f"Watching log file: {log_file}")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Read the file in real-time
        with open(log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.1)  # Small delay to avoid busy waiting
                    
    except KeyboardInterrupt:
        print("\nStopped watching logs")

def view_pipe():
    """View logs from named pipe"""
    pipe_name = "transcriber_logs"
    
    if not os.path.exists(pipe_name):
        print(f"Named pipe '{pipe_name}' not found. Make sure the transcriber is running.")
        return
    
    print(f"Reading from named pipe: {pipe_name}")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        with open(pipe_name, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped reading from pipe")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pipe":
        view_pipe()
    else:
        view_log_file() 