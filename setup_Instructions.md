QUILL - Setup Instructions
Quick Start Guide
This guide will help you set up QUILL on your computer.

Prerequisites
You need Python 3.8 or newer installed on your computer.
Check if you have Python:

Open Command Prompt (Windows) or Terminal (Mac/Linux)
Type: python --version or python3 --version
You should see something like "Python 3.10.x"

Don't have Python?

Download from: https://www.python.org/downloads/
IMPORTANT: During installation on Windows, check the box "Add Python to PATH"


Installation Steps
Step 1: Extract the Files
Unzip the QUILL folder to a location you can easily find (like your Desktop or Documents folder).
Step 2: Open Terminal/Command Prompt in the QUILL Folder
Windows:

Open the QUILL folder in File Explorer
Click in the address bar at the top
Type cmd and press Enter

Mac/Linux:

Open Terminal
Type cd  (with a space after cd)
Drag the QUILL folder into the Terminal window
Press Enter

Step 3: Install Required Packages
Copy and paste this command, then press Enter:
Windows:
pip install -r requirements.txt
Mac/Linux:
pip3 install -r requirements.txt
This will install all necessary libraries. It may take 2-5 minutes.
Note: If you see errors about transformers or torch, that's okay! The grammar checking will still work. To enable AI features, run:
pip install transformers torch
(This is a large download - about 1-2 GB)
Step 4: Run QUILL
Once installation is complete, run:
Windows:
python main.py
Mac/Linux:
python3 main.py
The QUILL application window should open!

Troubleshooting
"python is not recognized as a command"

Python is not installed or not in your PATH
Try python3 instead of python
Reinstall Python and check "Add to PATH"

"No module named 'PyQt6'"

The requirements didn't install properly
Try running the pip install command again

The app opens but AI features don't work

This is normal if you didn't install transformers/torch
Grammar checking will still work fine
To enable AI: pip install transformers torch

"LanguageTool failed to load"

Close and reopen the app
Check your internet connection (LanguageTool downloads rules on first run)