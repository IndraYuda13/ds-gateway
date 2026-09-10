#!/usr/bin/env python3
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client.deepseek_cli import main

if __name__ == "__main__":
    main()
