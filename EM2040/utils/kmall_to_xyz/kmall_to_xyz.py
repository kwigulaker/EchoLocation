"""
A simplified version of the kmall-converter, for readability and usability as a cli.
The logic is moved to process_datagram.py
"""
import os
import sys
import struct
import time
import portalocker
import utm
import click
from glob import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial
# Own imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_functions"))
from process_datagram import processDatagram
from file_handling import check_exclusive_lock

click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("src", nargs=1)
@click.argument("dst", nargs=1)
@click.option("-e", "--ext", type=click.Choice([".csv", ".xyz"]), default=".csv", help="File extension.")
@click.option("--gen_folders", is_flag=True, help="Structure in folders per kmall-file.")
@click.option("-x", "--latitude", is_flag=True, default=False, help="Include x (latitude or utm easting) column.")
@click.option("-y", "--longitude", is_flag=True, default=False, help="Include y (longitude or utm northing) column.")
@click.option("-z", "--depth", is_flag=True, default=False, help="Include z (depth) column.")
@click.option("-r", "--reflection", is_flag=True, default=False, help="Include r (reflection) column.")
@click.option("-c", "--coordinate_system", type=click.Choice(["relative", "latlon", "utm"]), default="latlon", help="Selected coordinate system, which define the relationship between coordinates.")
@click.option('-t', '--timeout', default=60, type=int, help='Timeout between checks.')
@click.option("-l", '--logging_level', default=0, help="Level of verbosity. 0 includes bare minimum, 1 includes additional information and 2 includes data just for convenience.")
@click.option("--rm_pings", is_flag=True, default=False, help="Remove .pings-files.")
@click.option("--rm_kmall", is_flag=True, default=False, help="Remove .kmall-files.")
def main(src, dst, ext, gen_folders, latitude, longitude, depth, reflection, coordinate_system, timeout, logging_level, rm_pings, rm_kmall):

	# Print runtime information
	XYZ_DECIMALS = 8
	data_fields = f"{'x' if latitude else ''}{'y' if longitude else ''}{'z' if depth else ''}{'r' if reflection else ''}"
	if logging_level > 0:
		nl = '\n'
		click.echo(nl.join([
			"",
			"-"*10 + " Runtime info " + "-"*10,
			f"src: {src}",
			f"dst: {dst}",
			f"data: {data_fields}",
			f"structure in folders: {'Yes' if gen_folders else 'No'}",
			f"coordinate system: {'relative latlon' if coordinate_system == 'relative' else ''}{'EPSG:4326 (WGS 84) latlon' if coordinate_system == 'latlon' else ''}{'EPSG:32632 (WGS 84 / UTM zone 32N)' if coordinate_system == 'utm' else ''}",
			f"remove .pings-files: {'Yes' if rm_pings else 'No'}",
			f"remove .kmall-files: {'Yes' if rm_kmall else 'No'}",
			"-"*34,
			""
		]))
	
	# Fix paths
	src = Path(os.path.realpath(src))
	dst = Path(os.path.realpath(dst))

	# Run until stopped by ctrl + c, listening on folder and run when files appear
	try:
		while True:
			# Find files
			kmall_files_no_ext = set([k[:-6] for k in glob(str(src / '**/*.kmall'), recursive=True) if "9999.kmall" not in k]) # Remove their default log called 9999.kmall
			ping_files_no_ext = set([p[:-12] for p in glob(str(src / '**/*.pings'), recursive=True)])
			untracked_files_no_ext = kmall_files_no_ext - ping_files_no_ext
			untracked_files = [k + ".kmall" for k in untracked_files_no_ext if not check_exclusive_lock(k + ".kmall")]

			# Create ping files for xyz if untracked
			if untracked_files:
				print(f"Step 1: Extracting ping-information from {len(untracked_files)} .kmall-file{'' if len(untracked_files) == 1 else 's'}")
				for file in tqdm(untracked_files):
					
					# Open the file for writing
					try:
						kmallIO = open(file, 'rb')
					except Exception:
						print('File', file, 'not opened.')
						sys.exit(0)
				
					# Process the file:
					kmallIO.seek(0, 2)
					file_size = kmallIO.tell()
					kmallIO.seek(0, 0)
					remaining = file_size

					# Open out file
					outputIO = open(file + ".pings", "a", encoding="utf-8")

					# Read all datagrams and process each of them
					while (remaining > 0):
						# First read 4 bytes that contains the length of the chunk
						lengthb = struct.unpack("I", kmallIO.read(4))
						remaining -= 4
						# Then read the chunk.  Note that the length read includes the 4 bytes in the integer.
						dgmsize = lengthb[0] - 4
						chunk = kmallIO.read(dgmsize)
						remaining -= dgmsize
						# Then process this chunk
						processDatagram(dgmsize, chunk, outputIO)
					
					# Close files
					outputIO.close()
					kmallIO.close()

			# Create csv files from ping files
			ping_files_no_ext = [os.path.basename(p)[:-12] for p in glob(str(src / '**/*.pings'), recursive=True)]
			csv_files_all = [os.path.basename(c) for c in glob(str(dst / '**/*.csv'), recursive=True)]
			untracked_files = []
			for ping_base in ping_files_no_ext:
				if not any(f"{ping_base}.{coordinate_system}.{data_fields}" in csv for csv in csv_files_all):
					untracked_files.append(src / (ping_base + ".kmall.pings"))
			
			if untracked_files:
				print(f"Step 2: Converting file{'' if len(untracked_files) == 1 else 's'} from .kmall.pings to .csv")
				for file in tqdm(untracked_files):
					
					# Decide new path
					fname_no_ext = os.path.basename(file).split(".")[0]
					new_path = dst / f"{fname_no_ext}/{fname_no_ext}.{coordinate_system}.{data_fields}{ext}" if gen_folders else dst / (f"{fname_no_ext}.{coordinate_system}.{data_fields}{ext}")
				
					# Open files
					try:
						# Read ping file
						pingIO = open(file)
						
						# Open new file with exclusive lock
						os.makedirs(os.path.dirname(new_path), exist_ok=True) # If subfolders not exist
						outfile = open(new_path, 'w', encoding='utf-8')
						portalocker.lock(outfile, portalocker.LockFlags.EXCLUSIVE)
						
						# Write header
						header = ""
						if latitude:
							header += f"{'Latitude ' if coordinate_system == 'latlon' else 'Utm_easting ' if coordinate_system == 'utm' else 'X '}"
						if longitude:
							header += f"{'Longitude ' if coordinate_system == 'latlon' else 'Utm_northing ' if coordinate_system == 'utm' else 'Y '}"
						if depth:
							header += f"{'Z ' if coordinate_system == 'relative' else 'Depth '}"
						if reflection:
							header += f"{'R' if coordinate_system == 'relative' else 'Reflection'}"
						outfile.write(f"{header}\n")
					
					except Exception as e:
						print('File',file,'not opened.', e)
						sys.exit(0)

					# Read data from ping-file, where first row is metadata and next ones is data rows
					line = pingIO.readline()
					while (line):
						row = line.split()
						
						# Header line with metadata
						if (len(row) == 5):
							n = float(row[0])
							e = float(row[1])
							tide = 0
							toWlev = -float(row[3]) + tide
						
						# Data rows
						else:
							cnt = 0
							while(cnt < len(row)):
								
								# Coordinate X and Y based on coordinate system (relationship)
								x, y = float(row[cnt]), float(row[cnt + 1])
								lat, lon = n + x, e + y
								if coordinate_system == "latlon":
									x, y = lat, lon
								elif coordinate_system == "utm":
									x, y, zone_number, zone_letter = utm.from_latlon(lat, lon)

								# Depth being either regular depth or backscatter/reflection
								dpt = (float(row[cnt + 2]) + toWlev) * -1.0
								r = float(row[cnt + 3])

								# Decide which fields to write based on cli input
								# UTM X, UTM Y, negative depth and reflection
								fields = [
									f"{x:.{XYZ_DECIMALS}f}" if latitude else None,
									f"{y:.{XYZ_DECIMALS}f}"  if longitude else None,
									f"{dpt:.{2}f}"  if depth else None,
									f"{r:.{2}f}"  if reflection else None,
								]
								outputstr = " ".join([field for field in fields if field is not None]) + "\n"
								outfile.write(outputstr)			
								cnt += 4

						line = pingIO.readline()
					
					portalocker.unlock(outfile)
					outfile.close()
					
					# Handle cleanup
					if rm_pings:
						fname = str(file)
						if os.path.exists(fname):
							os.remove(fname) # Delete temporary file
					if rm_kmall:
						fname = str(file)[:-6]
						if os.path.exists(fname):
							os.remove(fname) # Delete .kmall file too
			else:
				if timeout == -1:
					break
				if logging_level > 1:
					print(f"Awaiting files, sleeping for {timeout} seconds")
				time.sleep(timeout)
				continue
                
	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
	main()