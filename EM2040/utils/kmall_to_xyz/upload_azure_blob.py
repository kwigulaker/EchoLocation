"""
AzureBlobFileHandler class, and cli implementation for uploading all images in folder to azure container.

Requires .env file in tree (the script looks for the file in parents) with the following field:
  AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=X;AccountName=X;AccountKey=X;EndpointSuffix=X"
"""
import os
import sys
import click
from glob import glob
from pathlib import Path
from functools import partial
from azure.storage.blob import BlobServiceClient, ContentSettings, ContainerClient
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class AzureBlobFileHandler:
  def __init__(self, azure_container:str):
    
    # Find connection string
    try:
      CONNECTION_STRING = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    except KeyError:
      print('AZURE_STORAGE_CONNECTION_STRING must be set in env')
      sys.exit(1)
    
    # Initialize
    self.azure_container = azure_container
    self.blob_service_client =  BlobServiceClient.from_connection_string(CONNECTION_STRING)
    self.container_client = ContainerClient.from_connection_string(CONNECTION_STRING, container_name=self.azure_container)

  def upload_all_images_in_folder(self, ext, local_img_path, azure_dir):
    local_img_path = Path(local_img_path)
    paths = [f for f in glob(str(local_img_path / f"*{ext}")) if os.path.isfile(f)]
 
    # Upload each file
    for file in paths:
      self.upload_image(file, azure_dir)
 
  def upload_image(self, file:str, file_name:str=None, azure_dir:str=None, logging_level:int=0):
    if not file_name:
      file_name = os.path.basename(file)
    # Create blob with same name as local file name
    if azure_dir:
      new_file_name = str(Path(azure_dir) / file_name)
    blob_client = self.blob_service_client.get_blob_client(container=self.azure_container, blob=new_file_name)
 
    # Create blob on storage, overwrite if it already exists!
    image_content_setting = ContentSettings(content_type='image/tiff')
    if logging_level > 1:
      print(f"uploading file - {new_file_name}")
    with open(file, "rb") as data:
      blob_client.upload_blob(data, overwrite=True, content_settings=image_content_setting)
 
  def list_dir(self, azure_dir:str):
    blobs = list(self.container_client.list_blobs())
    in_dir = []
    for blob in blobs:
      if azure_dir in blob["name"]:
        in_dir.append(blob)
    return in_dir
    

click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("src", nargs=1)
@click.option("-e", "--ext", type=click.Choice([".tif", ".tiff"]), default=".tiff", help="File extension.")
def main(src, ext):
  # Initialize class and upload all files of ext in folder
  azure_blob_file_handler = AzureBlobFileHandler(azure_container="seabed")
  azure_blob_file_handler.upload_all_images_in_folder(ext=ext, local_img_path=src, azure_dir="original")

if __name__ == "__main__":
  main()
