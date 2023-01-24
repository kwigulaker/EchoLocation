"""
Functions for the preprocessing steps, being hillshade or slope.
"""
import numpy as np
import rasterio as rio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# Own imports
from geotiff_handling import GeoReference, set_raster_metadata, write_geotiff

#################
##  Hillshade  ##
#################
def _hillshade(array:np.ndarray, azimuth:int, angle_altitude:int):
    azimuth = 360.0 - azimuth
    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x*x+y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) + np.cos((azimuthrad - np.pi / 2.0) - aspect)
    out_array = 255 * (shaded+1) / 2
    return out_array

def _normalize(image:np.ndarray):
    image = np.nan_to_num(image, nan=0)
    maxim = np.max(image)
    image /= maxim + 0.5
    return image

def hillshade(array:np.ndarray, azimuth:int, angle_altitude:int, normalize:bool=False):
    out_array = _hillshade(array, azimuth, angle_altitude)
    if normalize:
        out_array = _normalize(out_array)
    return out_array

#############
##  Slope  ##
#############
def _nan_to_zero(image:np.ndarray, remove_edges=False):
    """
    Image are now having val 0 as nodata, so we have to set gradient 0 to 0.001, and replace nan with nodata. 
    Edge will have inf, so we set that to black as well if remove_edges. 
    """
    lowest_val_not_zero = 0.001
    image[image == 0] = lowest_val_not_zero
    image[np.isnan(image)] = 0
    if remove_edges:
        image[np.isinf(image)] = lowest_val_not_zero
    return image

def slope(array:np.ndarray, cellsize=0.25, normalize:bool=False, remove_edges=False):
    px, py = np.gradient(array, cellsize, edge_order=1)
    array = np.sqrt(px ** 2 + py ** 2)
    if normalize:
        array = _nan_to_zero(array, remove_edges=remove_edges)
    return array


def preprocess(file:str, dest_path:str, pipeline:str="CIUS", georeference=None) -> tuple:
    # Get georeference, setup metadata and read raster band
    with rio.open(file) as raster_img:
        if raster_img.crs and georeference:
            assert raster_img.crs == georeference
        # Get georeference
        georef = GeoReference(transform=raster_img.transform, projection=raster_img.crs if raster_img.crs else georeference)
        # Fetch metadata from original geoTIFF
        metadata = set_raster_metadata(raster_img, georef=georef, nodata=np.nan, dtype=rio.float32)
        # Read raster band as array
        img = raster_img.read(1)
    
    # Turn into numpy array
    np_arr = np.nan_to_num(img, nan=-np.inf)

    if pipeline=="CIUS":
        np_arr = hillshade(np_arr, 315, 45, normalize=True)
    elif pipeline=="MKDebris":
        np_arr = slope(np_arr, normalize=True, remove_edges=True)
    
    # Write as geoTIFF with georeference and correct projection 
    write_geotiff(np_arr, dest_path, metadata)

    return dest_path, georef

if __name__ == "__main__":
    pass
    

    