# running the machine learning algorithms and visualizing the results.


import pandas as pd
from preprocessing import clean_data
from algorithms import clustering, classification
from visualization import create_graphs

# Load the data
destinations = pd.read_csv('data/destinations.csv')
travel_modes = pd.read_csv('data/travel_modes.csv')
trip_durations = pd.read_csv('data/trip_durations.csv')
travel_costs = pd.read_csv('data/travel_costs.csv')

# Clean and preprocess the data
destinations_clean = clean_data(destinations)
travel_modes_clean = clean_data(travel_modes)
trip_durations_clean = clean_data(trip_durations)
travel_costs_clean = clean_data(travel_costs)

# Run the machine learning algorithms
patterns = clustering(destinations_clean)
trends = classification(travel_modes_clean, trip_durations_clean, travel_costs_clean)

# Visualize the results
create_graphs(patterns, trends)
