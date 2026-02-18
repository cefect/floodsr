# example chip from 2407_FHIMP

extracted from `/home/cefect/LS/09_REPOS/02_JOBS/2407_FHIMP/misc/chip_extractor.py`

```python
python misc/chip_extractor.py \
    --sim-dir workflow_outdir/07_unpack2/1m_utm11_e_21_166/0100_015 \
    --timestamp 0935 \
    --chip-index 0
    
```

yielding two layers:



```python
layer '1m_utm11_e_21_166_0100_015_0935_chip00000_wsh032'
    source:/home/cefect/LS/09_REPOS/02_JOBS/2407_FHIMP/1m_utm11_e_21_166_0100_015_0935_chip00000_wsh032.tif
    bbox (xmin, ymin, xmax, ymax):
     -1300733.076761606, 429695.823186665, -1299709.076761606, 430719.823186665 [EPSG:3979]
    32 w x 32 h = 1024 (32.00000000, 32.00000000)
    bandCount (QGIS): 1
    IMAGE_STRUCTURE: {'COMPRESSION': 'LZW', 'INTERLEAVE': 'BAND'}, NoDataValue: -9999.0
    BLOCKSIZE:[32, 32]

 
 
layer '1m_utm11_e_21_166_0100_015_0935_chip00000_dem002'
    source:/home/cefect/LS/09_REPOS/02_JOBS/2407_FHIMP/1m_utm11_e_21_166_0100_015_0935_chip00000_dem002.tif
    bbox (xmin, ymin, xmax, ymax):
     -1300733.076761606, 429695.823186665, -1299709.076761606, 430719.823186665 [EPSG:3979]
    512 w x 512 h = 262144 (2.00000000, 2.00000000)
    bandCount (QGIS): 1
    IMAGE_STRUCTURE: {'COMPRESSION': 'LZW', 'INTERLEAVE': 'BAND'}, NoDataValue: -9999.0
    BLOCKSIZE:[512, 1]
    
```