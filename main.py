import sys
import os

# Enable High-DPI scaling awareness on Windows to prevent blurry/pixelated UI
if sys.platform.startswith("win"):
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Process_System_DPI_Aware
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tkinter import Tk
from gui import NutriVisionGUI


def main():
    root = Tk()
    app = NutriVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()