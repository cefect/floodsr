#!/usr/bin/env bash
set -euo pipefail

# Input rasters.
study='dudelange'
dem_hires_fp="_inputs/RSSHydro/${study}/002/DEM.tif"
depth_hires_fp="_inputs/RSSHydro/${study}/002/ResultA.tif"
depth_lores_fp="_inputs/RSSHydro/${study}/032/ResultA.tif"
out_dir="tests/data/rss_${study}_A"

# Output driver options mirrored from parameters.py: GEOTIF_OPTIONS.
gdal_driver="GTiff"
gdal_dtype="Float32"
gdal_compress="LZW"
gdal_nodata="-9999"

# Target low-resolution chip size in pixels.
lowres_chip_size_px=256

# Validate required inputs.
for fp in "${dem_hires_fp}" "${depth_hires_fp}" "${depth_lores_fp}"; do
    [[ -f "${fp}" ]] || { echo "missing input raster: ${fp}" >&2; exit 1; }
done
mkdir -p "${out_dir}"

# Resolve CRS from DEM; this CRS is assigned to lowres outputs.
dem_srs="$(gdalsrsinfo -o epsg "${dem_hires_fp}" | awk "NF { print; exit }" | tr -d "[:space:]")"
[[ -n "${dem_srs}" ]] || { echo "failed to derive DEM CRS from ${dem_hires_fp}" >&2; exit 1; }

# Read lowres geotransform to derive a deterministic upper-left 256x256 tile bbox.
lowres_gt_json="$(gdalinfo -json "${depth_lores_fp}")"
lowres_xmin="$(echo "${lowres_gt_json}" | jq -r ".geoTransform[0]")"
lowres_ymax="$(echo "${lowres_gt_json}" | jq -r ".geoTransform[3]")"
lowres_xres_abs="$(echo "${lowres_gt_json}" | jq -r ".geoTransform[1] | if . < 0 then -. else . end")"
lowres_yres_abs="$(echo "${lowres_gt_json}" | jq -r ".geoTransform[5] | if . < 0 then -. else . end")"
lowres_res_int="$(awk -v v="${lowres_xres_abs}" "BEGIN { printf \"%d\", int(v + 0.5) }")"
lowres_y_res_int="$(awk -v v="${lowres_yres_abs}" "BEGIN { printf \"%d\", int(v + 0.5) }")"
[[ "${lowres_res_int}" -eq "${lowres_y_res_int}" ]] || { echo "non-square lowres resolution: ${lowres_xres_abs}, ${lowres_yres_abs}" >&2; exit 1; }

# Read hires resolutions and assert consistency between DEM and depth hires.
dem_gt_json="$(gdalinfo -json "${dem_hires_fp}")"
depth_hires_gt_json="$(gdalinfo -json "${depth_hires_fp}")"
dem_xres_abs="$(echo "${dem_gt_json}" | jq -r ".geoTransform[1] | if . < 0 then -. else . end")"
dem_yres_abs="$(echo "${dem_gt_json}" | jq -r ".geoTransform[5] | if . < 0 then -. else . end")"
depth_hires_xres_abs="$(echo "${depth_hires_gt_json}" | jq -r ".geoTransform[1] | if . < 0 then -. else . end")"
depth_hires_yres_abs="$(echo "${depth_hires_gt_json}" | jq -r ".geoTransform[5] | if . < 0 then -. else . end")"
hires_res_int="$(awk -v v="${dem_xres_abs}" "BEGIN { printf \"%d\", int(v + 0.5) }")"
dem_y_res_int="$(awk -v v="${dem_yres_abs}" "BEGIN { printf \"%d\", int(v + 0.5) }")"
depth_hires_x_res_int="$(awk -v v="${depth_hires_xres_abs}" "BEGIN { printf \"%d\", int(v + 0.5) }")"
depth_hires_y_res_int="$(awk -v v="${depth_hires_yres_abs}" "BEGIN { printf \"%d\", int(v + 0.5) }")"
[[ "${hires_res_int}" -eq "${dem_y_res_int}" ]] || { echo "non-square DEM resolution: ${dem_xres_abs}, ${dem_yres_abs}" >&2; exit 1; }
[[ "${hires_res_int}" -eq "${depth_hires_x_res_int}" && "${hires_res_int}" -eq "${depth_hires_y_res_int}" ]] || {
    echo "hires raster resolutions differ between DEM and depth" >&2
    exit 1
}

# Build output file names using the same convention as tests/data/2407_FHIMP_tile.
lowres_suffix="$(printf "%03d" "${lowres_res_int}")"
hires_suffix="$(printf "%03d" "${hires_res_int}")"
lowres_out_fp="${out_dir}/lowres${lowres_suffix}.tif"
depth_hires_out_fp="${out_dir}/hires${hires_suffix}.tif"
dem_hires_out_fp="${out_dir}/hires${hires_suffix}_dem.tif"

# Compute lowres chip bbox from upper-left origin and integer resolution.
tile_width_m="$(awk -v r="${lowres_res_int}" -v n="${lowres_chip_size_px}" "BEGIN { printf \"%.8f\", r * n }")"
lowres_xmax="$(awk -v x="${lowres_xmin}" -v w="${tile_width_m}" "BEGIN { printf \"%.8f\", x + w }")"
lowres_ymin="$(awk -v y="${lowres_ymax}" -v w="${tile_width_m}" "BEGIN { printf \"%.8f\", y - w }")"

# 1) Clip lowres to 256x256, while assigning CRS from the DEM.
gdalwarp -overwrite \
    -of "${gdal_driver}" \
    -ot "${gdal_dtype}" \
    -co "COMPRESS=${gdal_compress}" \
    -srcnodata "${gdal_nodata}" \
    -dstnodata "${gdal_nodata}" \
    -r near \
    -s_srs "${dem_srs}" \
    -t_srs "${dem_srs}" \
    -te "${lowres_xmin}" "${lowres_ymin}" "${lowres_xmax}" "${lowres_ymax}" \
    -tr "${lowres_res_int}" "${lowres_res_int}" \
    "${depth_lores_fp}" "${lowres_out_fp}"

# Resolve exact lowres output bbox to drive aligned clipping of both hires rasters.
lowres_out_info_json="$(gdalinfo -json "${lowres_out_fp}")"
bbox_xmin="$(echo "${lowres_out_info_json}" | jq -r ".cornerCoordinates.upperLeft[0]")"
bbox_ymax="$(echo "${lowres_out_info_json}" | jq -r ".cornerCoordinates.upperLeft[1]")"
bbox_xmax="$(echo "${lowres_out_info_json}" | jq -r ".cornerCoordinates.lowerRight[0]")"
bbox_ymin="$(echo "${lowres_out_info_json}" | jq -r ".cornerCoordinates.lowerRight[1]")"

# 2) Clip hires depth to lowres bbox at fixed 2m resolution with nearest-neighbor resampling.
gdalwarp -overwrite \
    -of "${gdal_driver}" \
    -ot "${gdal_dtype}" \
    -co "COMPRESS=${gdal_compress}" \
    -srcnodata "${gdal_nodata}" \
    -dstnodata "${gdal_nodata}" \
    -r near \
    -s_srs "${dem_srs}" \
    -t_srs "${dem_srs}" \
    -te "${bbox_xmin}" "${bbox_ymin}" "${bbox_xmax}" "${bbox_ymax}" \
    -tr "${hires_res_int}" "${hires_res_int}" \
    "${depth_hires_fp}" "${depth_hires_out_fp}"

# 3) Clip hires DEM to the same bbox/grid as hires depth for direct pixel alignment.
gdalwarp -overwrite \
    -of "${gdal_driver}" \
    -ot "${gdal_dtype}" \
    -co "COMPRESS=${gdal_compress}" \
    -srcnodata "${gdal_nodata}" \
    -dstnodata "${gdal_nodata}" \
    -r near \
    -t_srs "${dem_srs}" \
    -te "${bbox_xmin}" "${bbox_ymin}" "${bbox_xmax}" "${bbox_ymax}" \
    -tr "${hires_res_int}" "${hires_res_int}" \
    "${dem_hires_fp}" "${dem_hires_out_fp}"

echo "Wrote:"
echo "  ${lowres_out_fp}"
echo "  ${depth_hires_out_fp}"
echo "  ${dem_hires_out_fp}"
