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

Output GeoTIFF bands
--------------------
    1. fire_mask      uint8 stored as float32  – 0-9 classification
    2. algorithm_qa   uint8 stored as float32  – per-pixel QA bitfield
    3. confidence     uint8 stored as float32  – detection confidence (0-100 %)
    4. frp            float32 (MW)             – NaN where no fire
    5. bt_i4          float32 (K)              – I4 brightness temp, NaN where no fire

Authentication
--------------
Uses the `earthaccess` library, which handles NASA Earthdata OAuth automatically.
Credentials are read from (in order):
  1. Environment variables  EARTHDATA_USER  and  EARTHDATA_PASS
  2. ~/.netrc entry for urs.earthdata.nasa.gov
  3. Interactive prompt at runtime (saved to .netrc for future use)

Dependencies
------------
  pip install earthaccess netCDF4 h5py numpy scipy pyresample rasterio

Usage
-----
  # Download a granule by date + time (UTC HHMM):
  python vnp14img_to_geotiff.py --date 2024-08-15 --time 1542 --outdir ./output

  # Restrict to a bounding box (lon_min lat_min lon_max lat_max):
  python vnp14img_to_geotiff.py \\
      --date 2024-08-15 --time 1542 \\
      --bbox -125 30 -100 50 \\
      --outdir ./output

  # Use files already on disk (skip download entirely):
  python vnp14img_to_geotiff.py \\
      --fire-file  VNP14IMG.A2024228.1542.002.*.nc \\
      --geo-file   VNP03MODLL.A2024228.1542.002.*.h5 \\
      --outdir ./output
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import rasterio
from pyresample import geometry, kd_tree
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# CONSTANTS
# ===========================================================================

SINU_R     = 6_371_007.181
PIXEL_SIZE = 375.0
PROJ_SINU  = f"+proj=sinu +R={SINU_R} +nadgrids=@null +wktext"
SINU_CRS   = CRS.from_proj4(PROJ_SINU)

FIRE_MASK_CLASSES = {
    0: "not processed (non-zero QF)",  1: "bowtie",
    2: "sun glint",                    3: "water",
    4: "cloud",                        5: "clear land",
    6: "unclassified fire",            7: "low confidence fire",
    8: "nominal confidence fire",      9: "high confidence fire",
}

BAND_META = [
    # (dict_key,       np_dtype,    nodata,  long_name)
    ("fire_mask",    np.uint8,   255,    "Fire Mask (classes 0-9)"),
    ("algorithm_qa", np.uint8,   255,    "Algorithm QA bitfield (low 8 bits)"),
    ("confidence",   np.uint8,   255,    "Detection confidence (0-100 %)"),
    ("frp",          np.float32, np.nan, "Fire Radiative Power (MW)"),
    ("bt_i4",        np.float32, np.nan, "I4 Brightness Temperature (K)"),
]


# ===========================================================================
# 1.  AUTHENTICATION  (earthaccess)
# ===========================================================================

def get_earthaccess_session():
    """
    Authenticate with NASA Earthdata via earthaccess and return an
    authenticated requests.Session suitable for LP DAAC HTTPS downloads.

    earthaccess checks (in order):
      - EARTHDATA_USER / EARTHDATA_PASS environment variables
      - ~/.netrc  (machine urs.earthdata.nasa.gov)
      - Interactive prompt (and optionally saves to ~/.netrc)
    """
    try:
        import earthaccess
    except ImportError:
        sys.exit(
            "ERROR: earthaccess is not installed.\n"
            "Install it with:  pip install earthaccess\n"
            "Then re-run the script."
        )

    log.info("Authenticating with NASA Earthdata via earthaccess …")

    # earthaccess.login raises LoginAttemptFailure (not just returns False)
    # when credentials are wrong, so we must catch it and try the next strategy.
    auth = None
    for strategy, kwargs in [
        ("environment", {}),
        ("netrc",       {}),
        ("interactive", {"persist": True}),
    ]:
        try:
            log.info("  Trying strategy: %s", strategy)
            auth = earthaccess.login(strategy=strategy, **kwargs)
            if auth.authenticated:
                break
        except Exception as e:
            log.warning("  Strategy '%s' failed: %s", strategy, e)

    if auth is None or not auth.authenticated:
        sys.exit(
            "ERROR: All Earthdata auth strategies failed.\n"
            "Run this in a Python shell to save credentials to ~/.netrc:\n"
            "  import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
        )

    log.info("  Authenticated as: %s", auth.username)
    return earthaccess.get_requests_https_session()


# ===========================================================================
# 2.  CMR GRANULE SEARCH  (via earthaccess)
# ===========================================================================

def search_granules(short_name, date_str, hhmm, bbox=None, version="002"):
    """
    Search NASA CMR for granules near a specific date + time (+-3/+9 min window).
    Returns a list of earthaccess DataGranule objects (may be empty).
    """
    import earthaccess
    import datetime

    dt_str = f"{date_str}T{hhmm[:2]}:{hhmm[2:]}:00"
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
    t0 = dt - datetime.timedelta(minutes=3)
    t1 = dt + datetime.timedelta(minutes=9)
    temporal = (t0.strftime("%Y-%m-%dT%H:%M:%SZ"), t1.strftime("%Y-%m-%dT%H:%M:%SZ"))

    kwargs = dict(short_name=short_name, version=version, temporal=temporal, count=5)
    if bbox:
        kwargs["bounding_box"] = tuple(bbox)

    log.info("Searching CMR: %s  %s -> %s  bbox=%s", short_name, temporal[0], temporal[1], bbox)
    results = earthaccess.search_data(**kwargs)
    log.info("  -> %d granule(s) found", len(results))
    return results


def list_granules(date_str, end_date_str=None, bbox=None, version="002"):
    """
    List all VNP14IMG granules for a full day (or date range) + optional bbox.
    Prints a table so you can pick the right --time value.
    """
    import earthaccess

    if end_date_str is None:
        end_date_str = date_str
    temporal = (f"{date_str}T00:00:00Z", f"{end_date_str}T23:59:59Z")
    kwargs = dict(short_name="VNP14IMG", version=version, temporal=temporal, count=300)
    if bbox:
        kwargs["bounding_box"] = tuple(bbox)

    log.info("Listing VNP14IMG granules  %s -> %s  bbox=%s", temporal[0], temporal[1], bbox)
    results = earthaccess.search_data(**kwargs)

    if not results:
        print("\nNo VNP14IMG granules found.")
        print("Tips:")
        print("  - Try a wider bbox, or remove --bbox to see all global granules for the day")
        print("  - VIIRS passes over any given area only ~2x per day")
        return []

    print(f"\n{'#':<4}  {'Granule title':<52}  {'Start (UTC)':<22}  End (UTC)")
    print("-" * 108)
    for i, g in enumerate(results):
        title  = g["umm"].get("GranuleUR", "?")[:52]
        tr     = g["umm"].get("TemporalExtent", {}).get("RangeDateTime", {})
        t0_str = tr.get("BeginningDateTime", "?")
        t1_str = tr.get("EndingDateTime", "?")
        print(f"{i:<4}  {title:<52}  {t0_str:<22}  {t1_str}")

    print(f"\nTotal: {len(results)} granule(s)")
    print("\nRe-run with --time HHMM using the UTC start time shown above.")
    print("Example: if start is 2024-08-15T20:12:00Z  ->  --time 2012")
    return results


# ===========================================================================
# 3.  DOWNLOAD  (via earthaccess)
# ===========================================================================

def download_granule(granule, dest_dir, session):
    """
    Download all data files for a single earthaccess DataGranule.
    Returns the first downloaded Path (the .nc or .h5 file).

    earthaccess.download() handles Bearer-token auth, redirects, and
    progress reporting automatically.
    """
    import earthaccess

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading: %s", granule["meta"]["concept-id"])
    # earthaccess.download returns a list of local file paths
    local_paths = earthaccess.download([granule], local_path=str(dest_dir))

    if not local_paths:
        raise RuntimeError(f"earthaccess.download returned no files for {granule}")

    paths = [Path(p) for p in local_paths]
    log.info("  -> %s", [p.name for p in paths])
    return paths[0]   # return the primary data file


# ===========================================================================
# 4.  READ VNP03MODLL (750 m geolocation, HDF5)
# ===========================================================================

def read_vnp03modll(geo_path):
    """
    Read Latitude/Longitude from VNP03MODLL and upsample 2x to 375 m.

    VNP03MODLL is at 750 m (M-band):  shape [nscans*16, 3200]
    After 2x bilinear zoom:           shape [nscans*32, 6400]  <- matches VNP14IMG
    """
    log.info("Reading geolocation: %s", Path(geo_path).name)

    with h5py.File(geo_path, "r") as hf:
        # LP DAAC V002 standard path
        candidate_paths = [
            "/geolocation_data/Latitude",
            "/Latitude",
            "/latitude",
            "/Geolocation_Fields/Latitude",
        ]
        lat_path = next((p for p in candidate_paths if p in hf), None)

        if lat_path is None:
            # Walk file as last resort
            found = []
            hf.visititems(lambda n, o: found.append(n)
                          if isinstance(o, h5py.Dataset) and "atitude" in n else None)
            if not found:
                raise KeyError(
                    f"Cannot find Latitude in {geo_path}. "
                    f"Root keys: {list(hf.keys())}"
                )
            lat_path = "/" + found[0]

        lon_path = (lat_path
                    .replace("Latitude", "Longitude")
                    .replace("latitude", "longitude"))

        lats_750 = hf[lat_path][:].astype(np.float32)
        lons_750 = hf[lon_path][:].astype(np.float32)

    log.info("  VNP03MODLL shape (750 m): %s", lats_750.shape)

    # Mask fill values before interpolating
    lats_750[(lats_750 < -90)  | (lats_750 > 90)]  = np.nan
    lons_750[(lons_750 < -180) | (lons_750 > 180)] = np.nan

    # Bilinear 2x upsample
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

    Returns dict with:
      fire_mask, algorithm_qa    : 2-D uint8/uint32 arrays
      FP_line, FP_sample         : 1-D row/col indices of fire pixels
      FP_power, FP_T4, FP_T5    : 1-D float arrays (per fire pixel)
      FP_confidence, FP_latitude, FP_longitude, FP_day  : 1-D arrays
    """
    log.info("Reading fire product: %s", Path(fire_path).name)
    data = {}

    with nc.Dataset(fire_path, "r") as ds:
        # 2-D image layers (variable names contain spaces in V002)
        for nc_name, key, dtype in [
            ("fire mask",    "fire_mask",    np.uint8),
            ("algorithm QA", "algorithm_qa", np.uint32),
        ]:
            if nc_name in ds.variables:
                arr = np.asarray(ds.variables[nc_name][:])
                if hasattr(arr, "filled"):
                    arr = arr.filled(0)
                data[key] = arr.astype(dtype)
            else:
                log.warning("Variable '%s' not found.", nc_name)

        # 1-D sparse fire-pixel arrays
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

    fm = data.get("fire_mask", np.array([]))
    log.info("  Fire mask: %s  |  fire pixels: %d",
             fm.shape, len(data.get("FP_latitude", [])))
    return data


# ===========================================================================
# 6.  RECONSTRUCT 2-D LAYERS FROM SPARSE FP_* ARRAYS
# ===========================================================================

def _scatter(fire_data, fp_key, shape, fill):
    """Place 1-D FP values into a 2-D grid using FP_line / FP_sample."""
    arr2d = np.full(shape, fill,
                    dtype=np.float32 if isinstance(fill, float) else np.uint8)
    lines   = fire_data.get("FP_line")
    samples = fire_data.get("FP_sample")
    values  = fire_data.get(fp_key)
    if lines is None or samples is None or values is None:
        return arr2d
    nrows, ncols = shape
    rows = np.clip(lines.astype(int),   0, nrows - 1)
    cols = np.clip(samples.astype(int), 0, ncols - 1)
    arr2d[rows, cols] = values.astype(arr2d.dtype)
    return arr2d

def build_confidence_layer(fd): return _scatter(fd, "FP_confidence", fd["fire_mask"].shape, np.uint8(0))
def build_frp_layer(fd):        return _scatter(fd, "FP_power",      fd["fire_mask"].shape, np.nan)
def build_bt_layer(fd):         return _scatter(fd, "FP_T4",         fd["fire_mask"].shape, np.nan)


# ===========================================================================
# 7.  OUTPUT AREA DEFINITION  (sinusoidal, swath-sized)
# ===========================================================================

def make_area_def(lats, lons, pixel_size=PIXEL_SIZE):
    """Compute a tight sinusoidal AreaDefinition covering the swath footprint."""
    import pyproj
    proj  = pyproj.Proj(PROJ_SINU)
    valid = np.isfinite(lats) & np.isfinite(lons)
    if not valid.any():
        raise ValueError("No valid geolocation in swath.")

    x, y = proj(lons[valid], lats[valid])
    x_min = float(x.min()) - pixel_size
    x_max = float(x.max()) + pixel_size
    y_min = float(y.min()) - pixel_size
    y_max = float(y.max()) + pixel_size

    width  = max(1, int(round((x_max - x_min) / pixel_size)))
    height = max(1, int(round((y_max - y_min) / pixel_size)))
    log.info("Output area: %d x %d px", width, height)

    return geometry.AreaDefinition(
        area_id="sinu_375m", description="VNP14IMG 375m Sinusoidal",
        proj_id="sinu", projection=PROJ_SINU,
        width=width, height=height,
        area_extent=(x_min, y_min, x_max, y_max),
    )


# ===========================================================================
# 8.  RESAMPLE
# ===========================================================================

def resample_band(data, swath_def, area_def, fill_value, roi=600.0):
    in_data = data.astype(np.float64) if np.issubdtype(data.dtype, np.floating) else data
    return kd_tree.resample_nearest(
        swath_def, in_data, area_def,
        radius_of_influence=roi, fill_value=fill_value,
        epsilon=0.5, nprocs=1,
    )


# ===========================================================================
# 9.  WRITE GEOTIFF
# ===========================================================================

def write_geotiff(resampled, area_def, out_path):
    x_min, y_min, x_max, y_max = area_def.area_extent
    transform = from_origin(x_min, y_max, PIXEL_SIZE, PIXEL_SIZE)
    height, width = area_def.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Writing %s  (%d bands, %dx%d)", out_path.name, len(BAND_META), width, height)

    with rasterio.open(out_path, "w",
                       driver="GTiff", crs=SINU_CRS, transform=transform,
                       width=width, height=height, count=len(BAND_META),
                       dtype="float32", nodata=np.nan,
                       compress="lzw", tiled=True,
                       blockxsize=512, blockysize=512, bigtiff="IF_SAFER") as dst:

        for i, (key, np_dtype, nodata, long_name) in enumerate(BAND_META, start=1):
            arr = resampled.get(key)
            if arr is None:
                arr = np.full((height, width), np.nan, dtype=np.float32)
            else:
                arr = arr.astype(np.float32)
                if nodata == 255:
                    arr[arr == 255] = np.nan   # map integer nodata -> NaN

            dst.write(arr, i)
            dst.update_tags(i, long_name=long_name,
                            original_dtype=np_dtype.__name__,
                            original_nodata=str(nodata))

        dst.update_tags(
            PRODUCT="VNP14IMG.002", GEO_PRODUCT="VNP03MODLL.002",
            PROJECTION=f"Sinusoidal +R={SINU_R}", PIXEL_SIZE_M=str(PIXEL_SIZE),
            FIRE_MASK_CLASSES=str(FIRE_MASK_CLASSES),
        )

    log.info("  -> %.1f MB", out_path.stat().st_size / 1e6)
    return out_path


# ===========================================================================
# 10.  PIPELINE
# ===========================================================================

def run_pipeline(fire_file, geo_file, out_dir, roi=600.0):
    """Run the full swath-to-grid pipeline for one granule pair."""
    fire_file, geo_file, out_dir = Path(fire_file), Path(geo_file), Path(out_dir)

    # Read
    lats, lons = read_vnp03modll(geo_file)
    fire_data  = read_vnp14img(fire_file)

    # Align shapes (tiny row mismatches at granule edges)
    fm = fire_data["fire_mask"]
    r  = min(lats.shape[0], fm.shape[0])
    c  = min(lats.shape[1], fm.shape[1])
    if (lats.shape[0], lats.shape[1]) != (fm.shape[0], fm.shape[1]):
        log.warning("Shape mismatch geo=%s fire=%s -> cropping to (%d,%d)",
                    lats.shape, fm.shape, r, c)
    lats = lats[:r, :c];  lons = lons[:r, :c]
    for k in ("fire_mask", "algorithm_qa"):
        if k in fire_data:
            fire_data[k] = fire_data[k][:r, :c]

    # Build 2-D layers
    conf_2d = build_confidence_layer(fire_data)
    frp_2d  = build_frp_layer(fire_data)
    bt_2d   = build_bt_layer(fire_data)

    # Pyresample geometries
    swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
    area_def  = make_area_def(lats, lons)

    # Resample
    log.info("Resampling 5 bands …")
    def _r(arr, fill): return resample_band(arr, swath_def, area_def, fill, roi)
    resampled = {
        "fire_mask":    _r(fire_data.get("fire_mask",    np.zeros_like(fm)), 255),
        "algorithm_qa": _r(fire_data.get("algorithm_qa", np.zeros_like(fm)), 255),
        "confidence":   _r(conf_2d,  255),
        "frp":          _r(frp_2d,   np.nan),
        "bt_i4":        _r(bt_2d,    np.nan),
    }

    # Write
    stem     = fire_file.stem
    out_path = out_dir / f"{stem}_sinu_375m.tif"
    return write_geotiff(resampled, area_def, out_path)


# ===========================================================================
# 11.  DOWNLOAD WRAPPER
# ===========================================================================

def download_and_run(date_str, hhmm, out_dir, bbox=None, roi=600.0):
    """Search CMR, download a granule pair, run the pipeline."""
    session = get_earthaccess_session()
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"

    fire_granules = search_granules("VNP14IMG",   date_str, hhmm, bbox=bbox)
    geo_granules  = search_granules("VNP03MODLL", date_str, hhmm, bbox=bbox)

    if not fire_granules:
        raise RuntimeError(f"No VNP14IMG granules for {date_str} {hhmm}")
    if not geo_granules:
        raise RuntimeError(f"No VNP03MODLL granules for {date_str} {hhmm}")

    log.info("Using granules: %s  +  %s",
             fire_granules[0]["meta"]["concept-id"],
             geo_granules[0]["meta"]["concept-id"])

    fire_file = download_granule(fire_granules[0], raw_dir, session)
    geo_file  = download_granule(geo_granules[0],  raw_dir, session)

    return run_pipeline(fire_file, geo_file, out_dir, roi)


# ===========================================================================
# 12.  CLI
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="VNP14IMG + VNP03MODLL -> 375m sinusoidal GeoTIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g1 = p.add_argument_group("Local files (skip download)")
    g1.add_argument("--fire-file", metavar="PATH", help="VNP14IMG *.nc on disk")
    g1.add_argument("--geo-file",  metavar="PATH", help="VNP03MODLL *.h5 on disk")

    g2 = p.add_argument_group("Discover / download from NASA CMR  (requires earthaccess)")
    g2.add_argument("--date",     metavar="YYYY-MM-DD", help="Acquisition date (required)")
    g2.add_argument("--end-date", metavar="YYYY-MM-DD",
                    help="End date for --list range (default: same as --date)")
    g2.add_argument("--time",     metavar="HHMM", default=None,
                    help=(
                        "Granule start time UTC, e.g. 2012.  "
                        "Run --list first to see available times. "
                        "Required for download; ignored with --list."
                    ))
    g2.add_argument("--bbox",     metavar="COORD", nargs=4, type=float,
                    help="Spatial filter: lon_min lat_min lon_max lat_max")
    g2.add_argument("--list",     action="store_true",
                    help=(
                        "List available granules for --date [--end-date] [--bbox] "
                        "without downloading. Use this to find the right --time value."
                    ))

    p.add_argument("--outdir",  metavar="DIR",    default="./output")
    p.add_argument("--radius",  metavar="METRES", type=float, default=600.0,
                   help="Nearest-neighbour search radius in metres (default: 600)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bbox = tuple(args.bbox) if args.bbox else None

    # ------------------------------------------------------------------
    # Mode 1: list granules (no download)
    # ------------------------------------------------------------------
    if args.list:
        if not args.date:
            sys.exit("--list requires --date.")
        # Auth needed to search cloud-hosted collections
        get_earthaccess_session()
        list_granules(
            date_str=args.date,
            end_date_str=args.end_date,
            bbox=bbox,
        )
        return

    # ------------------------------------------------------------------
    # Mode 2: local files
    # ------------------------------------------------------------------
    if args.fire_file and args.geo_file:
        out = run_pipeline(args.fire_file, args.geo_file, args.outdir, args.radius)
        print(f"\nSuccess!  Output GeoTIFF:\n  {Path(out).resolve()}")
        return

    # ------------------------------------------------------------------
    # Mode 3: download from CMR
    # ------------------------------------------------------------------
    if not args.date:
        sys.exit("Provide --fire-file + --geo-file  OR  --date [--time].  Run with -h for help.")

    if not args.time:
        sys.exit(
            "ERROR: --time HHMM is required for download.\n"
            f"Run with --list --date {args.date}"
            + (f" --bbox {' '.join(str(v) for v in args.bbox)}" if args.bbox else "")
            + "  to see available granule times."
        )

    out = download_and_run(
        date_str=args.date,
        hhmm=args.time.replace(":", ""),
        out_dir=args.outdir,
        bbox=bbox,
        roi=args.radius,
    )
    print(f"\nSuccess!  Output GeoTIFF:\n  {Path(out).resolve()}")


if __name__ == "__main__":
    main()
