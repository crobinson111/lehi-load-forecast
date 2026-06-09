# Lehi Power Load Forecast — Setup Instructions

## What you need first

1. **Visual Studio Code** — download free at https://code.visualstudio.com
2. **Python 3.11 or later** — download free at https://www.python.org
   - During install, check **"Add Python to PATH"**
3. **Claude Code extension for VS Code**
   - Open VS Code, click the Extensions icon on the left sidebar (looks like four squares)
   - Search for **Claude Code** and install it
   - Sign in with your Anthropic or claude.ai account when prompted

---

## One-time setup

1. **Unzip** the `lehi-load-forecast` folder I sent you and put it somewhere convenient (like your Desktop or Documents)

2. **Open the folder in VS Code**
   - In VS Code: File → Open Folder → select the `lehi-load-forecast` folder

3. **Open Claude Code** — press `Ctrl+Shift+P`, type `Claude`, and select **Claude: Open** (or click the Claude icon in the sidebar)

4. **Paste this message into Claude Code and hit Enter:**

   > Set up this project for me. Install the Python dependencies from requirements.txt, then create a .env file with my Excel file path. My file is at: **[paste your full file path here]**

   Claude will install everything and configure your data source automatically.

---

## Every time you want to use the app

1. Open the `lehi-load-forecast` folder in VS Code
2. Open Claude Code and paste this message:

   > Start the load forecast app

   Claude will launch the server. When it says "Uvicorn running", open your browser to **http://localhost:8000**

3. The model trains automatically on startup (takes about 60–90 seconds). Once the status dot turns green, pick a date and click **Get Report**.

4. **To stop the app**, go back to VS Code and press `Ctrl+C` in the terminal.

---

## The one thing you need to keep updated

Open `school_calendar.csv` in VS Code and check that the school break dates are correct each August when Alpine School District publishes its new calendar. You can ask Claude to help update it:

> Update school_calendar.csv with the new Alpine School District calendar for the 2026-27 school year: [paste the break dates]

---

## Troubleshooting

If anything goes wrong, open Claude Code and describe what happened. Claude can read the project files and fix most issues automatically.
