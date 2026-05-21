# Import the required libraries
import json
from pydantic import BaseModel

# Define a class to represent the shipment schema
class Shipment(BaseModel):
    # Shipper name
    shipper_name: str
    # Consignee
    consignee: str
    # Origin city
    origin_city: str
    # Destination city
    destination_city: str
    # Weight in kg
    weight_in_kg: int
    # Container number
    container_number: str
    # Estimated delivery date
    estimated_delivery_date: str

# Read the sample shipment file
def read_shipment_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the file content
        content = file.read()
        # Return the content
        return content

# Extract shipment data from the file content
def extract_shipment_data(content):
    # Split the content into lines
    lines = content.split('\n')
    # Initialize an empty dictionary to store the shipment data
    shipment_data = {}
    # Iterate over each line
    for line in lines:
        # Check if the line is not empty
        if line:
            # Split the line into key and value
            key, value = line.split(': ')
            # Store the key-value pair in the shipment data dictionary
            shipment_data[key] = value
    # Return the shipment data
    return shipment_data

# Create a shipment object from the extracted data
def create_shipment_object(shipment_data):
    # Create a new shipment object
    shipment = Shipment(
        shipper_name=shipment_data['Shipper Name'],
        consignee=shipment_data['Consignee'],
        origin_city=shipment_data['Origin City'],
        destination_city=shipment_data['Destination City'],
        weight_in_kg=int(shipment_data['Weight in kg']),
        container_number=shipment_data['Container Number'],
        estimated_delivery_date=shipment_data['Estimated Delivery Date']
    )
    # Return the shipment object
    return shipment

# Print the shipment data as JSON
def print_shipment_data(shipment):
    # Convert the shipment object to JSON
    shipment_json = shipment.model_dump_json()
    # Print the JSON
    print(shipment_json)

# Save the shipment data to a file
def save_shipment_data(shipment):
    # Convert the shipment object to JSON
    shipment_json = shipment.model_dump_json()
    # Open the output file in write mode
    with open('parsed_shipment.json', 'w') as file:
        # Write the JSON to the file
        json.dump(json.loads(shipment_json), file, indent=4)

# Main function
def main():
    # Read the sample shipment file
    file_content = read_shipment_file('sample_shipment.txt')
    # Extract the shipment data
    shipment_data = extract_shipment_data(file_content)
    # Create a shipment object
    shipment = create_shipment_object(shipment_data)
    # Print the shipment data as JSON
    print_shipment_data(shipment)
    # Save the shipment data to a file
    save_shipment_data(shipment)

# Call the main function
if __name__ == '__main__':
    main()