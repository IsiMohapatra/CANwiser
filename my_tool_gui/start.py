import sys
import os

# Ensure parent directory is in sys.path for package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_tool_gui.main_app import CANwiserApp

if __name__ == "__main__":
    app = CANwiserApp()
    app.mainloop()