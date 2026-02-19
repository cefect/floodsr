"""project wide parameters"""
from pathlib import Path

 

PROJECT_ROOT = Path(__file__).parent.resolve()
 

GEOTIF_OPTIONS= dict(driver='GTiff',
                     dtype='float32',
                        compress='LZW',
                        nodata=-9999, 
                        )

GPKG_OPTIONS= dict(driver='GPKG', engine='pyogrio',
                      dataset_options={"VERSION": "1.4"},
                      layer_options={"GEOMETRY_NAME": "geometry", "FID": "fid", "SPATIAL_INDEX": "YES",
                                     #geometry precision handling
                                     "DISCARD_COORD_LSB":"YES", "UNDO_DISCARD_COORD_LSB_ON_READING":"YES",})
 