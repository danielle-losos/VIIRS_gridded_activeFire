"""
VNP14IMG → 375 m Sinusoidal GeoTIFF Pipeline
==============================================
Downloads a single VNP14IMG (375 m active fire swath) granule and its
companion VNP03MODLL (750 m geolocation) granule from NASA LP DAAC,
reprojects the fire mask (and selected auxiliary bands) onto a
375 m sinusoidal grid, and writes the result as a multi-band GeoTIFF.

Products used
-------------
  VNP14IMG.002   – VIIRS/NPP Active Fires 6-Min L2 Swath 375m V002
                   https://doi.org/10.5067/VIIRS/VNP14IMG.002
  VNP03MODLL.002 – VIIRS/NPP Moderate Resolution Terrain Corrected
                   Geolocation 6-Min L1 Swath 750m Light V002
                   https://doi.org/10.5067/VIIRS/VNP03MODLL.002

Key design decisions
--------------------
* NASA officially prescribes VNP03MODLL (750 m) as the geolocation
  companion for VNP14IMG.  VNP03MODLL lat/lon arrays are at 750 m so
  half the density of the fire mask.  They are upsampled 2x with
  bilinear interpolation before passing to pyresample.
* Output projection: MODIS/VIIRS Sinusoidal (+proj=sinu +R=6371007.181),
  375 m pixel, same grid family as VNP14A1 but at 375 m instead of 1 km.
* Output GeoTIFF bands (in order):
    1. fire_mask      (uint8)    – 0-9 classification
    2. algorithm_qa   (uint8)    – per-pixel QA bitfield (low 8 bits)
    3. confidence     (uint8)    – detection confidence (0-100 %)
    4. frp            (float32)  – Fire Radiative Power, MW  (NaN = no fire)
    5. bt_i4          (float32)  – I4 brightness temperature, K (NaN = no fire)

Authentication
--------------
Requires a NASA Earthdata account.  Set credentials in one of three ways
(checked in this order):
  1. Environment variables  EARTHDATA_USER  and  EARTHDATA_PASS
  2. ~/.netrc entry for urs.earthdata.nasa.gov
  3. Prompted interactively at runtime

Dependencies
------------
  pip install requests netCDF4 h5py numpy scipy pyresample rasterio

Usage
-----
  # Download a granule by date + time (UTC):
  python vnp14img_to_geotiff.py --date 2024-08-15 --time 0000 --outdir ./output

  # Spatial filter (lon_min lat_min lon_max lat_max):
  python vnp14img_to_geotiff.py \\
      --date 2024-08-15 --time 0000 \\
      --bbox -125 30 -100 50 \\
      --outdir ./output

  # Skip download – use files already on disk:
  python vnp14img_to_geotiff.py \\
      --fire-file  VNP14IMG.A2024228.0000.002.nc \\
      --geo-file   VNP03MODLL.A2024228.0000.002.h5 \\
      --outdir ./output
"""

import argparse
import getpass
import logging
import netrc
import os
import re
import sys
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import requests
import rasterio
from pyresample import geometry, kd_tree
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  CONSTANTS
# ===========================================================================

SINU_R     = 6_371_007.181          # MODIS/VIIRS sinusoidal sphere radius (m)
PIXEL_SIZE = 375.0                  # output pixel size (m)
PROJ_SINU  = f"+proj=sinu +R={SINU_R} +nadgrids=@null +wktext"
SINU_CRS   = CRS.from_proj4(PROJ_SINU)

CMR_SEARCH = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Fire mask classification (for GeoTIFF metadata)
FIRE_MASK_CLASSES = {
    0: "not processed (non-zero QF)",
    1: "bowtie",
    2: "sun glint",
    3: "water",
    4: "cloud",
    5: "clear land",
    6: "unclassified fire",
    7: "low confidence fire",
    8: "nominal confidence fire",
    9: "high confidence fire",
}

# GeoTIFF band definitions: (dict_key, np_dtype, nodata, long_name)
BAND_META = [
    ("fire_mask",    np.uint8,   255,    "Fire Mask (classes 0-9)"),
    ("algorithm_qa", np.uint8,   255,    "Algorithm QA bitfield (low 8 bits)"),
    ("confidence",   np.uint8,   255,    "Detection confidence (0-100 %)"),
    ("frp",          np.float32, np.nan, "Fire Radiative Power (MW)"),
    ("bt_i4",        np.float32, np.nan, "I4 Brightness Temperature (K)"),
]


# ===========================================================================
# 2.  AUTHENTICATION
# ===========================================================================

def get_credentials(user=None, password=None):
    """Return (username, password) from args, env vars, .netrc, or prompt."""
    if user and password:
        return user, password
    user = os.environ.get("EARTHDATA_USER")
    password = os.environ.get("EARTHDATA_PASS")
    if user and password:
        log.info("Earthdata credentials from environment variables.")
        return user, password
    try:
        n = netrc.netrc()
        auth = n.authenticators("urs.earthdata.nasa.gov")
        if auth:
            log.info("Earthdata credentials from ~/.netrc.")
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    log.info("No stored credentials – prompting interactively.")
    user = input("NASA Earthdata username: ")
    password = getpass.getpass("NASA Earthdata password: ")
    return user, password


def build_session(username, password):
    """
    Build a requests.Session pre-loaded with Earthdata credentials.
    The session follows URS OAuth redirects automatically (max 10 hops).
    """
    session = requests.Session()
    session.auth = (username, password)
    session.max_redirects = 10
    return session


# ===========================================================================
# 3.  CMR GRANULE SEARCH AND DOWNLOAD
# ===========================================================================

def search_granules(short_name, date_str, hhmm, bbox=None,
                    version="002", provider="LPCLOUD", page_size=5):
    """
    Query NASA CMR for granules matching product + date/time window.

    Parameters
    ----------
    short_name : str   e.g. "VNP14IMG"
    date_str   : str   "YYYY-MM-DD"
    hhmm       : str   "HHMM" UTC e.g. "1542"
    bbox       : tuple (lon_min, lat_min, lon_max, lat_max) or None
    version    : str   collection version
    provider   : str   CMR provider (LPCLOUD = LP DAAC cloud archive)
    page_size  : int   max results to return

    Returns
    -------
    list of CMR granule entry dicts (may be empty)
    """
    import datetime
    dt_str = f"{date_str}T{hhmm[:2]}:{hhmm[2:]}:00Z"
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    # ±3 min window to account for slight time differences between products
    t0 = (dt - datetime.timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = (dt + datetime.timedelta(minutes=9)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "short_name": short_name,
        "version":    version,
        "provider":   provider,
        "temporal":   f"{t0},{t1}",
        "page_size":  page_size,
    }
    if bbox:
        params["bounding_box"] = ",".join(str(v) for v in bbox)

    log.info("CMR search %s  temporal=[%s, %s]  bbox=%s", short_name, t0, t1, bbox)
    r = requests.get(CMR_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", [])
    log.info("  -> %d granule(s) found", len(entries))
    return entries


def pick_download_url(granule):
    """
    Extract the best HTTPS data URL from a CMR granule entry dict.
    Preference order:
      1. LP DAAC cloud protected HTTPS link
      2. Any 'data#' rel link
      3. Any .nc / .h5 HTTPS link
    """
    links = granule.get("links", [])
    for lnk in links:
        href = lnk.get("href", "")
        if "lp-prod-protected" in href and href.startswith("https://"):
            return href
    for lnk in links:
        if "data#" in lnk.get("rel", "") and lnk.get("href", "").startswith("https://"):
            return lnk["href"]
    for lnk in links:
        href = lnk.get("href", "")
        if href.startswith("https://") and re.search(r"\.(nc|h5)$", href, re.I):
            return href
    raise ValueError(
        f"No suitable download URL in granule '{granule.get('title')}'. "
        f"Links: {[l.get('href') for l in links]}"
    )


def download_file(url, dest_dir, session):
    """
    Download url into dest_dir, skipping if the file already exists.
    Returns the local Path.
    """
    fname = Path(url.split("?")[0]).name
    dest  = Path(dest_dir) / fname
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already on disk: %s", dest.name)
        return dest
    log.info("Downloading %s …", fname)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    log.info("  Saved %.1f MB -> %s", dest.stat().st_size / 1e6, dest)
    return dest


# ===========================================================================
# 4.  READ VNP03MODLL (750 m geolocation, HDF5)
# ===========================================================================

def read_vnp03modll(geo_path):
    """
    Read Latitude / Longitude from a VNP03MODLL HDF5 file and upsample
    2x to match the 375 m VNP14IMG grid.

    VNP03MODLL stores geolocation at 750 m (M-band resolution):
        /geolocation_data/Latitude    float32  [nscans*16, 3200]
        /geolocation_data/Longitude   float32  [nscans*16, 3200]

    After 2x bilinear zoom the arrays become [nscans*32, 6400],
    matching VNP14IMG's fire mask dimensions.

    Returns
    -------
    lats_375, lons_375 : float32 arrays shape (nrows_375m, 6400)
    """
    log.info("Reading geolocation: %s", Path(geo_path).name)

    with h5py.File(geo_path, "r") as hf:
        # Find lat/lon – LP DAAC V002 uses /geolocation_data/ group
        candidate_lat_paths = [
            "/geolocation_data/Latitude",
            "/Latitude",
            "/latitude",
            "/Geolocation_Fields/Latitude",
        ]
        lat_path = None
        for cp in candidate_lat_paths:
            if cp in hf:
                lat_path = cp
                break
        if lat_path is None:
            # Last resort: walk the file
            def _find(name, obj):
                if isinstance(obj, h5py.Dataset) and "atitude" in name:
                    return name
            lat_path = hf.visititems(_find)
        if lat_path is None:
            raise KeyError(
                f"Cannot find Latitude in {geo_path}. "
                f"Root keys: {list(hf.keys())}"
            )
        lon_path = lat_path.replace("Latitude", "Longitude").replace("atitude", "ongitude")

        lats_750 = hf[lat_path][:].astype(np.float32)
        lons_750 = hf[lon_path][:].astype(np.float32)

    log.info("  VNP03MODLL shape (750 m): %s", lats_750.shape)

    # Replace fill values with NaN before interpolating
    lats_750[(lats_750 < -90)  | (lats_750 > 90)]  = np.nan
    lons_750[(lons_750 < -180) | (lons_750 > 180)] = np.nan

    # Bilinear 2x upsample via scipy.ndimage.zoom (order=1)
    # NaN handling: zoom propagates NaN cleanly with order=1
    lats_375 = zoom(lats_750, zoom=2, order=1, mode="nearest")
    lons_375 = zoom(lons_750, zoom=2, order=1, mode="nearest")

    log.info("  Upsampled to 375 m: %s", lats_375.shape)
    return lats_375, lons_375


# ===========================================================================
# 5.  READ VNP14IMG (375 m active fire, NetCDF4)
# ===========================================================================

def read_vnp14img(fire_path):
    """
    Read VNP14IMG V002 NetCDF4 file.

    Returns a dict containing:
      2-D image arrays (shape = fire mask grid):
        fire_mask     : uint8
        algorithm_qa  : uint32

      1-D sparse arrays (one entry per fire pixel):
        FP_latitude, FP_longitude,
        FP_line, FP_sample,        <- row/col index in the 2-D grid
        FP_power, FP_T4, FP_T5,
        FP_confidence, FP_day
    """
    log.info("Reading fire product: %s", Path(fire_path).name)
    data = {}

    with nc.Dataset(fire_path, "r") as ds:
        # 2-D image arrays – V002 NetCDF variable names include a space
        for nc_name, key, dtype in [
            ("fire mask",   "fire_mask",    np.uint8),
            ("algorithm QA","algorithm_qa", np.uint32),
        ]:
            if nc_name in ds.variables:
                arr = np.asarray(ds.variables[nc_name][:])
                if hasattr(arr, "filled"):
                    arr = arr.filled(0)
                data[key] = arr.astype(dtype)
            else:
                log.warning("Variable '%s' not found – will be skipped.", nc_name)

        # 1-D sparse fire pixel arrays
        for vname in [
            "FP_latitude", "FP_longitude",
            "FP_line",     "FP_sample",
            "FP_power",    "FP_T4",   "FP_T5",
            "FP_confidence", "FP_day",
        ]:
            if vname in ds.variables:
                arr = np.asarray(ds.variables[vname][:])
                if hasattr(arr, "filled"):
                    fill = np.nan if arr.dtype.kind == "f" else 0
                    arr = arr.filled(fill)
                data[vname] = arr
            else:
                log.debug("Optional variable '%s' absent.", vname)

    fm = data.get("fire_mask", np.array([]))
    log.info("  Fire mask shape: %s  |  fire pixels: %d",
             fm.shape, len(data.get("FP_latitude", [])))
    return data


# ===========================================================================
# 6.  RECONSTRUCT 2-D LAYERS FROM SPARSE FP_* ARRAYS
# ===========================================================================

def _scatter_fp_to_2d(fire_data, fp_key, shape, fill_value):
    """
    Place 1-D sparse FP values back into a 2-D image using
    FP_line (row) and FP_sample (col) as index arrays.
    """
    arr_2d = np.full(shape, fill_value,
                     dtype=np.float32 if isinstance(fill_value, float) else np.uint8)
    lines   = fire_data.get("FP_line")
    samples = fire_data.get("FP_sample")
    values  = fire_data.get(fp_key)
    if lines is None or samples is None or values is None:
        return arr_2d
    nrows, ncols = shape
    rows = np.clip(lines.astype(int),  0, nrows - 1)
    cols = np.clip(samples.astype(int), 0, ncols - 1)
    arr_2d[rows, cols] = values.astype(arr_2d.dtype)
    return arr_2d


def build_confidence_layer(fire_data):
    shape = fire_data["fire_mask"].shape
    return _scatter_fp_to_2d(fire_data, "FP_confidence", shape, fill_value=np.uint8(0))


def build_frp_layer(fire_data):
    shape = fire_data["fire_mask"].shape
    return _scatter_fp_to_2d(fire_data, "FP_power", shape, fill_value=np.nan)


def build_bt_layer(fire_data):
    shape = fire_data["fire_mask"].shape
    return _scatter_fp_to_2d(fire_data, "FP_T4", shape, fill_value=np.nan)


# ===========================================================================
# 7.  DEFINE 375 m SINUSOIDAL OUTPUT AREA
# ===========================================================================

def make_area_def(lats, lons, pixel_size=PIXEL_SIZE):
    """
    Build a pyresample AreaDefinition that tightly covers the swath extent
    in sinusoidal projection.

    The area is computed dynamically so that the output GeoTIFF covers
    only the swath footprint rather than the full global grid.

    Parameters
    ----------
    lats, lons  : 2-D float32 arrays  (degrees)
    pixel_size  : output pixel size (m)

    Returns
    -------
    pyresample.geometry.AreaDefinition
    """
    import pyproj
    proj = pyproj.Proj(PROJ_SINU)

    valid = np.isfinite(lats) & np.isfinite(lons)
    if not valid.any():
        raise ValueError("No valid geolocation pixels in swath.")

    x_arr, y_arr = proj(lons[valid], lats[valid])
    x_min = float(x_arr.min()) - pixel_size
    x_max = float(x_arr.max()) + pixel_size
    y_min = float(y_arr.min()) - pixel_size
    y_max = float(y_arr.max()) + pixel_size

    width  = max(1, int(round((x_max - x_min) / pixel_size)))
    height = max(1, int(round((y_max - y_min) / pixel_size)))

    log.info(
        "Output area: %d x %d px  x=[%.0f, %.0f]  y=[%.0f, %.0f]",
        width, height, x_min, x_max, y_min, y_max,
    )

    return geometry.AreaDefinition(
        area_id="vnp14img_sinu_375m",
        description="VNP14IMG 375m Sinusoidal",
        proj_id="sinu",
        projection=PROJ_SINU,
        width=width,
        height=height,
        area_extent=(x_min, y_min, x_max, y_max),
    )


# ===========================================================================
# 8.  RESAMPLE
# ===========================================================================

def resample_band(data, swath_def, area_def,
                  fill_value, radius_of_influence=600.0):
    """
    Nearest-neighbour resample a 2-D swath band to area_def.

    radius_of_influence : search radius in metres
        600 m = 1.6 × 375 m pixel; catches the nearest swath pixel
        without introducing artefacts from distant mismatches.
    """
    # kd_tree requires float64 for float data; integers can stay as-is
    in_data = data.astype(np.float64) if np.issubdtype(data.dtype, np.floating) else data
    return kd_tree.resample_nearest(
        swath_def,
        in_data,
        area_def,
        radius_of_influence=radius_of_influence,
        fill_value=fill_value,
        epsilon=0.5,    # allow slight sub-optimality for speed
        nprocs=1,
    )


# ===========================================================================
# 9.  WRITE GEOTIFF
# ===========================================================================

def write_geotiff(resampled, area_def, out_path):
    """
    Write a multi-band Cloud-Optimised GeoTIFF from the resampled dict.

    Bands (see BAND_META):
        1  fire_mask     uint8
        2  algorithm_qa  uint8
        3  confidence    uint8
        4  frp           float32  (NaN = no fire / fill)
        5  bt_i4         float32  (NaN = no fire / fill)
    """
    x_min, y_min, x_max, y_max = area_def.area_extent
    transform = from_origin(x_min, y_max, PIXEL_SIZE, PIXEL_SIZE)
    height, width = area_def.shape
    n_bands = len(BAND_META)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Writing %s  (%d bands, %dx%d px)", out_path.name, n_bands, width, height)

    # Use separate nodata per band via a mixed-type approach:
    # write integer bands first (uint8 with 255 nodata) and float bands
    # separately is tricky with a single-file profile.
    # Solution: write all bands as float32, encoding uint8 bands as float32.
    # For cleanliness we write two separate GeoTIFFs and then merge,
    # OR we use the simplest approach: write float32 throughout and note
    # nodata per band in metadata tags.  Downstream tools handle this fine.

    profile = dict(
        driver="GTiff",
        crs=SINU_CRS,
        transform=transform,
        width=width,
        height=height,
        count=n_bands,
        dtype="float32",
        nodata=np.nan,          # NaN as universal fill (works for float32)
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        bigtiff="IF_SAFER",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for band_idx, (key, np_dtype, nodata, long_name) in enumerate(BAND_META, start=1):
            arr = resampled.get(key)
            if arr is None:
                log.warning("Band %d (%s) missing – filling with NaN.", band_idx, key)
                arr = np.full((height, width), np.nan, dtype=np.float32)
            else:
                arr = arr.astype(np.float32)
                # Map integer nodata (255) to NaN for uniform storage
                if nodata == 255:
                    arr[arr == 255] = np.nan
            dst.write(arr, band_idx)
            dst.update_tags(band_idx,
                            long_name=long_name,
                            original_nodata=str(nodata),
                            original_dtype=np_dtype.__name__)

        dst.update_tags(
            PRODUCT="VNP14IMG.002",
            GEO_PRODUCT="VNP03MODLL.002",
            PROJECTION="Sinusoidal +R=6371007.181",
            PIXEL_SIZE_M=str(PIXEL_SIZE),
            FIRE_MASK_CLASSES=str(FIRE_MASK_CLASSES),
            NOTE=(
                "Bands 1-3 are integer classes stored as float32 for "
                "uniformity; NaN = unobserved / fill."
            ),
        )

    log.info("  -> %.1f MB", out_path.stat().st_size / 1e6)
    return out_path


# ===========================================================================
# 10.  MAIN PIPELINE
# ===========================================================================

def run_pipeline(fire_file, geo_file, out_dir, radius_of_influence=600.0):
    """
    Complete swath-to-grid pipeline for one granule pair.

    Parameters
    ----------
    fire_file  : path-like  VNP14IMG *.nc file
    geo_file   : path-like  VNP03MODLL *.h5 file
    out_dir    : path-like  output directory
    radius_of_influence : pyresample search radius (m)

    Returns
    -------
    Path of the written GeoTIFF
    """
    fire_file = Path(fire_file)
    geo_file  = Path(geo_file)
    out_dir   = Path(out_dir)

    # ------------------------------------------------------------------
    # A. Read inputs
    # ------------------------------------------------------------------
    lats_375, lons_375 = read_vnp03modll(geo_file)
    fire_data          = read_vnp14img(fire_file)

    # ------------------------------------------------------------------
    # B. Align grid shapes
    # VNP03MODLL [M, 3200] at 750 m  ->  after 2x zoom: [2M, 6400] at 375 m
    # VNP14IMG fire mask is         [2M, 6400]  (same swath, same granule)
    # Tiny mismatches (1-2 rows) can occur at granule edges; crop to min.
    # ------------------------------------------------------------------
    fire_mask    = fire_data["fire_mask"]
    fm_rows, fm_cols = fire_mask.shape
    geo_rows, geo_cols = lats_375.shape

    if (geo_rows, geo_cols) != (fm_rows, fm_cols):
        log.warning(
            "Grid mismatch: geo=%s  fire=%s  ->  cropping to minimum.",
            (geo_rows, geo_cols), (fm_rows, fm_cols),
        )
        r = min(geo_rows, fm_rows)
        c = min(geo_cols, fm_cols)
        lats_375 = lats_375[:r, :c]
        lons_375 = lons_375[:r, :c]
        for key in ("fire_mask", "algorithm_qa"):
            if key in fire_data:
                fire_data[key] = fire_data[key][:r, :c]

    # ------------------------------------------------------------------
    # C. Reconstruct 2-D layers from 1-D sparse FP arrays
    # ------------------------------------------------------------------
    conf_2d = build_confidence_layer(fire_data)
    frp_2d  = build_frp_layer(fire_data)
    bt_2d   = build_bt_layer(fire_data)

    # ------------------------------------------------------------------
    # D. Define pyresample geometries
    # ------------------------------------------------------------------
    swath_def = geometry.SwathDefinition(lons=lons_375, lats=lats_375)
    area_def  = make_area_def(lats_375, lons_375, PIXEL_SIZE)

    # ------------------------------------------------------------------
    # E. Resample all bands
    # ------------------------------------------------------------------
    log.info("Resampling 5 bands to 375 m sinusoidal …")

    def _r(arr, fill):
        return resample_band(arr, swath_def, area_def, fill, radius_of_influence)

    resampled = {
        "fire_mask":    _r(fire_data.get("fire_mask",    np.zeros_like(fire_mask)), 255),
        "algorithm_qa": _r(fire_data.get("algorithm_qa", np.zeros_like(fire_mask)), 255),
        "confidence":   _r(conf_2d,  255),
        "frp":          _r(frp_2d,   np.nan),
        "bt_i4":        _r(bt_2d,    np.nan),
    }

    # ------------------------------------------------------------------
    # F. Write output GeoTIFF
    # ------------------------------------------------------------------
    stem     = fire_file.stem  # e.g. VNP14IMG.A2024228.0000.002.2024230120000
    out_path = out_dir / f"{stem}_sinu_375m.tif"
    write_geotiff(resampled, area_def, out_path)
    return out_path


# ===========================================================================
# 11.  DOWNLOAD WRAPPER
# ===========================================================================

def download_and_run(date_str, hhmm, out_dir,
                     bbox=None, username=None, password=None,
                     radius_of_influence=600.0):
    """
    Search CMR, download VNP14IMG + VNP03MODLL for the specified
    date/time, and run the pipeline.
    """
    user, pwd = get_credentials(username, password)
    session   = build_session(user, pwd)
    out_dir   = Path(out_dir)

    # Search
    fire_granules = search_granules("VNP14IMG",   date_str, hhmm, bbox=bbox)
    geo_granules  = search_granules("VNP03MODLL", date_str, hhmm, bbox=bbox)

    if not fire_granules:
        raise RuntimeError(f"No VNP14IMG granules found: {date_str} {hhmm} bbox={bbox}")
    if not geo_granules:
        raise RuntimeError(f"No VNP03MODLL granules found: {date_str} {hhmm} bbox={bbox}")

    fire_g = fire_granules[0]
    geo_g  = geo_granules[0]
    log.info("Fire granule: %s", fire_g.get("title"))
    log.info("Geo  granule: %s", geo_g.get("title"))

    # Download
    raw_dir   = out_dir / "raw"
    fire_file = download_file(pick_download_url(fire_g), raw_dir, session)
    geo_file  = download_file(pick_download_url(geo_g),  raw_dir, session)

    return run_pipeline(fire_file, geo_file, out_dir, radius_of_influence)


# ===========================================================================
# 12.  CLI
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Convert a VNP14IMG (375m active fire swath) + VNP03MODLL (geolocation) "
            "granule pair into a 375m sinusoidal GeoTIFF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )

    grp_local = p.add_argument_group("Local files (skip download)")
    grp_local.add_argument("--fire-file", metavar="PATH",
                           help="VNP14IMG *.nc file on disk")
    grp_local.add_argument("--geo-file",  metavar="PATH",
                           help="VNP03MODLL *.h5 file on disk")

    grp_dl = p.add_argument_group("Download from NASA CMR")
    grp_dl.add_argument("--date", metavar="YYYY-MM-DD",
                        help="Acquisition date (required for download)")
    grp_dl.add_argument("--time", metavar="HHMM", default="0000",
                        help="Acquisition time UTC, e.g. 1542 (default: 0000)")
    grp_dl.add_argument("--bbox", metavar="COORD", nargs=4, type=float,
                        help="lon_min lat_min lon_max lat_max")
    grp_dl.add_argument("--user",     metavar="USER",
                        help="Earthdata username (or EARTHDATA_USER env var)")
    grp_dl.add_argument("--password", metavar="PASS",
                        help="Earthdata password (or EARTHDATA_PASS env var)")

    p.add_argument("--outdir",  metavar="DIR",    default="./output",
                   help="Output directory (default: ./output)")
    p.add_argument("--radius",  metavar="METRES", type=float, default=600.0,
                   help="Nearest-neighbour search radius in metres (default: 600)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.fire_file and args.geo_file:
        out = run_pipeline(
            Path(args.fire_file), Path(args.geo_file),
            Path(args.outdir), args.radius,
        )
    elif args.date:
        out = download_and_run(
            date_str=args.date,
            hhmm=args.time.replace(":", ""),
            out_dir=Path(args.outdir),
            bbox=tuple(args.bbox) if args.bbox else None,
            username=args.user,
            password=args.password,
            radius_of_influence=args.radius,
        )
    else:
        sys.exit(
            "ERROR: Provide either --fire-file + --geo-file  OR  --date.\n"
            "Run with -h for help."
        )

    print(f"\nSuccess!  Output GeoTIFF:\n  {Path(out).resolve()}")


if __name__ == "__main__":
    main()
