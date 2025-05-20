import tkinter as tk
from tkinter import messagebox
import os

def real_time_detection():
    """Run the real-time mask detection script."""
    messagebox.showinfo("Info", "Starting real-time mask detection...")
    os.system("python real_time_mask_detection.py")

# Create the main application window
app = tk.Tk()
app.title("Real-Time Mask Detection")
app.geometry("400x200")

# Add button for real-time detection
tk.Label(app, text="Real-Time Mask Detection", font=("Arial", 16)).pack(pady=20)
tk.Button(app, text="Start Detection", command=real_time_detection, width=20, height=2).pack(pady=20)

# Run the application
app.mainloop()
