# fix_raw_data.py
import json

input_file = "data/raw/locomo10.json"
output_file = "data/raw/locomo10_fixed.json"

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Try to fix common JSON issues
# Remove trailing commas
import re
content = re.sub(r',\s*}', '}', content)
content = re.sub(r',\s*]', ']', content)

# Ensure it's a valid JSON array
if not content.strip().startswith('['):
    content = '[' + content
if not content.strip().endswith(']'):
    content = content + ']'

# Try to parse
try:
    data = json.loads(content)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Fixed data saved to {output_file}")
except json.JSONDecodeError as e:
    print(f"Could not fix automatically: {e}")
    print("Please check the raw data file format")