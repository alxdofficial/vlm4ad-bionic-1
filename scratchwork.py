import json
import os

# Define file paths
train_json_path = '/home/alex/Documents/vscodeprojects/personal/bionic driving vqa/data/multi_frame/multi_frame_train.json'
val_json_path = '/home/alex/Documents/vscodeprojects/personal/bionic driving vqa/data/multi_frame/multi_frame_val.json'
test_json_path = '/home/alex/Documents/vscodeprojects/personal/bionic driving vqa/data/multi_frame/multi_frame_test.json'

# Load JSON files
with open(train_json_path, 'r') as f:
    train_data = json.load(f)
with open(val_json_path, 'r') as f:
    val_data = json.load(f)
with open(test_json_path, 'r') as f:
    test_data = json.load(f)

# Combine all data
combined_data = train_data + val_data + test_data

# Extract timestamp from CAM_FRONT field
def extract_timestamp(item):
    cam_front_path = item[1]['CAM_FRONT']
    timestamp_str = cam_front_path.split('__')[-1].split('.')[0]
    timestamp = int(timestamp_str)
    return timestamp

# Sort combined data by timestamp
sorted_data = sorted(combined_data, key=extract_timestamp)

# Split data into train, val, and test sets
train_split = int(0.8 * len(sorted_data))
val_split = int(0.1 * len(sorted_data))

train_data = sorted_data[:train_split]
val_data = sorted_data[train_split:train_split + val_split]
test_data = sorted_data[train_split + val_split:]

# Create new directory for output
output_dir = '/home/alex/Documents/vscodeprojects/personal/bionic driving vqa/data/multi_frame_sorted'
os.makedirs(output_dir, exist_ok=True)

# Write new JSON files with pretty formatting
with open(os.path.join(output_dir, 'sorted_multi_frame_train.json'), 'w') as f:
    json.dump(train_data, f, indent=4)
with open(os.path.join(output_dir, 'sorted_multi_frame_val.json'), 'w') as f:
    json.dump(val_data, f, indent=4)
with open(os.path.join(output_dir, 'sorted_multi_frame_test.json'), 'w') as f:
    json.dump(test_data, f, indent=4)

print("Data sorted and split successfully.")
  