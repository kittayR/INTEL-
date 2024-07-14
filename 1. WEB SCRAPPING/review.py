import pandas as pd
import json

# Load the JSON data from a file
with open('B0BCF57FL5-reviews.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('4star-4.csv', index=False)