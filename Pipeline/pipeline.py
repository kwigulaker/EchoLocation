import os

# Define the input and output directories
input_dir = '../EM2040/data/clusters/moorings'
output_dir = '../EM2040/data/clusters/moorings_xyz'

# Make the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # Create the output filename by replacing the extension
        output_filename = os.path.splitext(filename)[0] + '.xyz'
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, output_filename)

        # Read the input file and convert to XYZ format
        with open(input_path, 'r') as f:
            lines = f.readlines()
            coords = []
            for line in lines[0:]:
                x, y, z = map(float, line.split())
                coords.append((x, y, z))

        # Write the XYZ file
        with open(output_path, 'w') as f:
            for x, y, z in coords:
                f.write(f'{x:.6f} {y:.6f} {z:.6f}\n')
