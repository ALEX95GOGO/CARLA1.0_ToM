import os
import numpy as np
import pandas as pd

# Define the number of consecutive time steps for trend analysis
consecutive_steps = 8
front_vehicle_brake_threshold = 0.3

# Create empty lists to store reaction times for each file
reaction_times = []

# Specify the folder path containing the CSV files
folder_path = r'G:\pilot_data\excel'

# Iterate through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Read the CSV file
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Extract data from the CSV
        front_vehicle_velocity = df['velocity_front']
        ego_vehicle_brake = df['brake']

        # Function to detect the start of braking
        def detect_ego_braking(velocity_data, brake_data, brake_threshold):
            # ... (same code as before) ...

        # Detect the start of braking
        front_brake_start_index, rear_brake_start_index = detect_ego_braking(front_vehicle_velocity, ego_vehicle_brake, front_vehicle_brake_threshold)

        if front_brake_start_index is not None and rear_brake_start_index is not None:
            # Calculate reaction time
            reaction_time = np.array(df['time'][rear_brake_start_index]) - np.array(df['time'][front_brake_start_index])
            reaction_times.append(reaction_time)
            print(f"Reaction time for {filename}: {np.mean(reaction_time)} seconds")
        else:
            print(f"No braking detected in {filename}.")

# Save the reaction times to a CSV file
if len(reaction_times) > 0:
    reaction_times_df = pd.DataFrame({'ReactionTime': reaction_times})
    reaction_times_df.to_csv('reaction_times.csv', index=False)
    print("Reaction times saved to reaction_times.csv")
else:
    print("No reaction times to save.")
