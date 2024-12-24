import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np

# Load the trained model and scaler from the .pkl files
with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# ----------------------
# Theme Configuration
# ----------------------
LIGHT_MODE = {
    "bg": "#E8F0FE",  # Light Blueish Background
    "fg": "#2C3E50",  # Dark Blue Text
    "entry_bg": "#FFFFFF",  # White Entry Fields
    "entry_fg": "#2C3E50",
    "button_bg": "#4CAF50",  # Green Button
    "button_fg": "#000000",
    "frame_bg": "#DDEEFF",  # Slightly Blue Background for Frames
}

DARK_MODE = {
    "bg": "#1E1E1E",  # Dark Grey Background
    "fg": "#FFFFFF",  # White Text
    "entry_bg": "#2C2C2C",  # Darker Entry Fields
    "entry_fg": "#FFFFFF",
    "button_bg": "#4CAF50",  # Green Button
    "button_fg": "#FFFFFF",
    "frame_bg": "#2A2A2A",  # Slightly Dark Background for Frames
}


# ----------------------
# Theme Toggle
# ----------------------
current_theme = LIGHT_MODE

def toggle_theme():
    global current_theme
    current_theme = DARK_MODE if current_theme == LIGHT_MODE else LIGHT_MODE
    
    root.config(bg=current_theme["bg"])
    title_label.config(bg=current_theme["bg"], fg=current_theme["fg"])
    footer_label.config(bg=current_theme["bg"], fg=current_theme["fg"])
    predict_button.config(bg=current_theme["button_bg"], fg=current_theme["button_fg"])
    theme_toggle_button.config(bg=current_theme["button_bg"], fg=current_theme["button_fg"])
    
    for frame, label, entry in zip(input_frames, input_labels, input_entries):
        frame.config(bg=current_theme["frame_bg"])
        label.config(bg=current_theme["frame_bg"], fg=current_theme["fg"])
        entry.config(bg=current_theme["entry_bg"], fg=current_theme["entry_fg"], insertbackground=current_theme["fg"])


# ----------------------
# GUI Initialization
# ----------------------
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("600x750")
root.config(bg=current_theme["bg"])


# ----------------------
# Title
# ----------------------
title_label = tk.Label(
    root, text="Heart Disease Prediction",
    font=("Helvetica", 20, "bold"), bg=current_theme["bg"], fg=current_theme["fg"]
)
title_label.pack(pady=20)


# ----------------------
# Input Fields
# ----------------------
frame = tk.Frame(root, bg=current_theme["frame_bg"], bd=2, relief="flat")
frame.pack(pady=10, padx=20, fill="both", expand=True)
frame.config(bd=0)

labels = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)", "Resting Blood Pressure",
    "Serum Cholesterol in mg/dl", "Fasting Blood Sugar (1=True, 0=False)", "Resting Electrocardiographic Results (0-2)",
    "Maximum Heart Rate Achieved", "Exercise Induced Angina (1=True, 0=False)", "Oldpeak (ST depression)",
    "Slope of Peak Exercise ST Segment", "Number of Major Vessels Colored by Fluoroscopy", "Thalassemia (0-3)"
]

entries = {}
input_frames = []
input_labels = []
input_entries = []

for label in labels:
    row_frame = tk.Frame(frame, bg=current_theme["frame_bg"], bd=0)
    row_frame.pack(fill='x', pady=8)
    input_frames.append(row_frame)
    
    label_widget = tk.Label(
        row_frame, text=label, font=("Helvetica", 10, "bold"),
        bg=current_theme["frame_bg"], fg=current_theme["fg"], anchor='w'
    )
    label_widget.pack(side='left', padx=10)
    input_labels.append(label_widget)
    
    entry_widget = tk.Entry(
        row_frame, bg=current_theme["entry_bg"], fg=current_theme["entry_fg"],
        font=("Helvetica", 10), insertbackground=current_theme["fg"],
        relief='flat', highlightthickness=1, highlightbackground="#B0BEC5", highlightcolor="#4CAF50"
    )
    entry_widget.pack(side='right', fill='x', expand=True, padx=10)
    input_entries.append(entry_widget)
    
    entries[label] = entry_widget


# ----------------------
# Prediction Function
# ----------------------
def predict():
    try:
        # Collect input data from user
        input_data = [float(entries[label].get()) for label in labels]
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        return

    # Rescale the input data using the saved scaler
    input_data_scaled = scaler.transform([input_data])

    # Make the prediction using the loaded model
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction == 1:
        result = "⚠️ The patient has heart disease."
    else:
        result = "✅ The patient does not have heart disease."

    # Show the result in a message box
    messagebox.showinfo("Prediction Result", result)


# ----------------------
# Buttons
# ----------------------
button_frame = tk.Frame(root, bg=current_theme["bg"])
button_frame.pack(pady=20)

predict_button = tk.Button(
    button_frame, text="Predict Heart Disease", command=predict,
    font=('Helvetica', 12, 'bold'), bg=current_theme["button_bg"], fg=current_theme["button_fg"],
    padx=10, pady=5, borderwidth=0, relief='flat', highlightthickness=1, highlightbackground="#388E3C"
)
predict_button.pack(pady=5)




# ----------------------
# Footer
# ----------------------
footer_label = tk.Label(
    root, text="© 2024 Heart Disease Predictor", font=("Helvetica", 10, "bold"),
    bg=current_theme["bg"], fg=current_theme["fg"]
)
footer_label.pack(pady=10)


# ----------------------
# Run GUI
# ----------------------
root.mainloop()
