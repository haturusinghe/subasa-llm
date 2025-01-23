import json

# Read the JSON file
with open('predictions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Now 'data' is a Python dictionary containing the JSON content
# If you have a list of objects, it will be a list of dictionaries

# Modifying the dataset
for obj in data:
    # Check if there's an offensive phrase in Predicted Labels
    if '\nOffensive Phrases: ' in obj['Predicted Labels']:
        # Split the string into label and offensive phrase
        label, offensive_phrase = obj['Predicted Labels'].split('\nOffensive Phrases: ')
        # Update the fields
        obj['Predicted Labels'] = label
        obj['Predicted Offensive Phrases'] = offensive_phrase

# Flatten the Rationale column
for obj in data:
    # Convert nested lists to 1D list by extracting the first element of each sublist
    obj['Rationale'] = [sublist[0] for sublist in obj['Rationale']]


# Save the modified dataset
with open('predictions_cleaned.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=1)


from sklearn.metrics import classification_report
import numpy as np

# Convert labels to binary format (OFF = 1, NOT = 0)
def convert_label(label):
    return 1 if label == 'OFF' else 0

# Create true and predicted label lists
y_true = [convert_label(obj['True Labels']) for obj in data]
y_pred = [convert_label(obj['Predicted Labels']) for obj in data]

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, 
                          target_names=['NOT', 'OFF'],
                          digits=4))

# save to a txt file
with open('classification_report.txt', 'w') as file:
    file.write(classification_report(y_true, y_pred, 
                          target_names=['NOT', 'OFF'],
                          digits=4))