import json
import csv
import argparse

def json_to_csv(input_file, output_file):
    # Read the JSON data
    with open(input_file, 'r') as f:
        json_data = f.read().splitlines()

    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(['input', 'output', 'avg_base_entropy', 'avg_expert_entropy', 'avg_antiexpert_entropy', 'avg_dexpert_entropy'])
        
        # Process each JSON object
        for line in json_data:
            try:
                data = json.loads(line)
                csvwriter.writerow([
                    data.get('input', ''),
                    data.get('output', ''),
                    data.get('avg_base_entropy', ''),
                    data.get('avg_expert_entropy', ''),
                    data.get('avg_antiexpert_entropy', ''),
                    data.get('avg_dexpert_entropy', '')
                ])
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")

    print(f"CSV file '{output_file}' has been created successfully.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert JSON file to CSV.')
    parser.add_argument('input_file', help='Input JSON file name')
    parser.add_argument('output_file', help='Output CSV file name')
    
    # Parse arguments
    args = parser.parse_args()

    # Call the conversion function with provided arguments
    json_to_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()