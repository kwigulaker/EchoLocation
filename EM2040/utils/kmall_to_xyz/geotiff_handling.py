"""
Functions for handling geotiffs, both in gdal and rasterio.
"""
import os
import numpy as np
import rasterio as rio
import rasterio.warp as riow
from osgeo import gdal, osr
from dataclasses import dataclass

PROJECTIONS = {
    "utm32n": "EPSG:32632", # Projected coordinate system for Between 6°E and 12°E, northern hemisphere between equator and 84°N, onshore and offshore.
    "wgs84": "EPSG:4326", # Geodetic coordinate system for World. Horizontal component of 3D system. Used by the GPS satellite navigation system and for NATO military.
    "worldmercator": "EPSG:3395", # Projected coordinate system for World between 80°S and 84°N. Euro-centric view of world excluding polar areas.
    "pseudomercator": "EPSG:3857" # Projected coordinate system for World between 85.06°S and 85.06°N. Uses spherical development of ellipsoidal coordinates. Relative to WGS 84.
}

#################################
##  Rasterio Geotiff handling  ##
#################################
@dataclass
class GeoReference:
    transform: str
    projection: str

def set_raster_metadata(raster:rio.io.DatasetWriter, georef:GeoReference=None, nodata=None, dtype:str=None):
    metadata = raster.meta.copy()
    if georef:
        metadata.update({
            'crs': georef.projection,
            'transform': georef.transform,
            'width': raster.width,
            'height': raster.height,
        })
    if nodata != None:
        metadata['nodata'] = nodata
    if dtype:
        metadata['dtype'] = dtype
    return metadata

def overwrite_raster_nodata(src:str, nodata=None):
    """Overwrites the geoinformation field 'nodata'.

    Args:
        src (str): File to change
        nodata (int/double/str, optional): New nodata value. Defaults to None.
    """
    assert os.path.isfile(src), "You have to send in a file!"
    with rio.open(src, "r+") as ds:
        if nodata != None:
            ds.nodata = nodata

def rewrite_raster_nodata(src:str, nodata=None):
    """Rewrites the geoinformation field 'nodata', in addition to replacing all nodata-values in the array.

    Args:
        src (str): File to change
        nodata (int/double/str, optional): New nodata value. Defaults to None.
    """
    assert os.path.isfile(src), "You have to send in a file!"
    with rio.open(src, "r+") as ds:
        if nodata != None:
            ds.nodata = nodata
        band = ds.read(1)
        band = np.nan_to_num(band, nan=0)
        ds.write(band, indexes=1)    
    

def write_geotiff(np_arr, dest_path, metadata) -> tuple:
    # Write as georeferenced raster
    with rio.open(dest_path, 'w', **metadata) as dst:
        dst.write_band(1, np_arr)


def rasterio_reproject(file:str, out_file:str, src_crs:str='EPSG:32632', dst_crs:str="EPSG:4326", nodata:int=None):
    assert file != out_file, "Overwrite not supported!"

    # Open file
    with rio.open(file) as raster:
    
        # Find EPSG of original file, and set if None
        rst_crs = raster.crs
        if rst_crs:
            assert rst_crs == src_crs, f"Raster projection {rst_crs}, but you defined {src_crs} as original coordinate system!"
            if rst_crs == dst_crs:
                print(f"Dst_crs ({dst_crs}) same as current projection ({rst_crs})!")
        else:
            rst_crs = src_crs

        # Calculate the transform matrix for the output, last argument unpacks outer boundaries (left, bottom, right, top)
        dst_transform, width, height = riow.calculate_default_transform(rst_crs, dst_crs, raster.width, raster.height, *raster.bounds)

        # set properties for output
        dst_kwargs = raster.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
            }
        )

        # Change nodata
        if nodata != None:
            dst_kwargs["nodata"] = nodata
            arr = np.nan_to_num(raster.read(1), nan=nodata)
        else:
            arr = rio.band(raster, 1)

        # Write file
        with rio.open(out_file, "w", **dst_kwargs) as dst:
            riow.reproject(
                source=arr,
                destination=rio.band(dst, 1),
                src_transform=raster.transform,
                src_crs=rst_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=riow.Resampling.nearest,
            )



#############################
##    GDAL reprojection    ##
#############################

def _gdal_find_epsg(raster):
    proj = osr.SpatialReference(wkt=raster.GetProjection())
    proj.AutoIdentifyEPSG()
    rst_crs = proj.GetAttrValue('AUTHORITY',1)
    return f"EPSG:{rst_crs}" if rst_crs else None

def gdal_reproject(file:str, out_file:str, src_crs:str='EPSG:32632', dst_crs:str="EPSG:4326"):
    assert file != out_file, "Overwrite not supported!"

    # Open file
    raster = gdal.Open(file)

    # Find EPSG of original file, and set if None
    rst_crs = _gdal_find_epsg(raster)
    if rst_crs:
        assert rst_crs == src_crs, f"Raster projection {rst_crs}, but you defined {src_crs} as original coordinate system!"
        if rst_crs == dst_crs:
            print(f"Dst_crs ({dst_crs}) same as current projection ({rst_crs})!")
    else:
        raster.SetProjection(src_crs)

    # Reproject using gdal
    warp = gdal.Warp(out_file, raster, dstSRS=dst_crs)
    warp = None
    
