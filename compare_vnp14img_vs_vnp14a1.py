"""
compare_vnp14img_vs_vnp14a1.py
==============================
Compare a VNP14IMG-derived 375m sinusoidal GeoTIFF (your swath product)
against the official VNP14A1 1km sinusoidal tile (L3 daily gridded product).

What this script does
---------------------
1. Reads VNP14A1 from its HDF-EOS5 tile (.h5), reconstructs the sinusoidal
   geotransform from the tile coordinates, and writes it as an in-memory
   rasterio dataset (or optionally a GeoTIFF).
2. Reprojects / resamples the 1km VNP14A1 tile to match the 375m GeoTIFF
   extent and grid exactly, so every pixel is directly comparable.
3. Computes per-class agreement statistics for the fire mask and a
   correlation / bias analysis for MaxFRP vs FRP.
4. Plots a 2×3 figure:
     Row 1: fire mask side-by-side (VNP14IMG 375m | VNP14A1 1km-upsampled)
     Row 2: difference map | scatter FRP | bar chart of class agreement

Usage
-----
  python compare_vnp14img_vs_vnp14a1.py \\
      --img   /path/to/VNP14IMG.A2020230.2036.002.*_sinu_375m.tif \\
      --l3    /path/to/VNP14A1.A2020230.h08v05.002.*.h5

  Optional flags:
    --save-l3-tif    write the reprojected VNP14A1 tile to disk as a GeoTIFF
    --outdir DIR     where to save figure + optional GeoTIFF (default: same dir as --img)
    --dpi N          figure DPI (default 150)

Dependencies
------------
  pip install h5py numpy rasterio matplotlib scipy

Fire mask class mapping
-----------------------
VNP14IMG (swath -> gridded by this pipeline):
  0  not processed / non-zero QF
  1  bowtie
  2  sun glint
  3  water
  4  cloud
  5  clear land
  6  unclassified fire
  7  low-confidence fire
  8  nominal-confidence fire
  9  high-confidence fire

VNP14A1 (L3 daily tile):
  0  missing input data
  1  not processed (trim)
  2  not processed (obsolete)
  3  non-fire water
  4  cloud
  5  non-fire land
  6  unknown
  7  fire (low confidence)
  8  fire (nominal confidence)
  9  fire (high confidence)

Classes 7-9 are "fire" in both products.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SINU_R     = 6_371_007.181
PROJ_SINU  = f"+proj=sinu +R={SINU_R} +nadgrids=@null +wktext"
SINU_CRS   = CRS.from_proj4(PROJ_SINU)

# Sinusoidal tile dimensions (MODIS/VIIRS standard)
TILE_SIZE_M  = 1_111_950.519667    # width/height of one tile in metres
TILE_NCOLS   = 2400                 # columns in a 500m tile... but VNP14A1 is 1km
# VNP14A1 is 1km → 1200×1200 per tile
L3_NROWS = L3_NCOLS = 1200
L3_PIXEL = TILE_SIZE_M / L3_NCOLS  # ≈ 926.625 m  (nominal "1km")

# Global sinusoidal origin (upper-left of tile h00v00)
SINU_X_ORIGIN = -20_015_109.354    # leftmost x
SINU_Y_ORIGIN =  10_007_554.677    # topmost  y  (north pole tile top)

FIRE_CLASSES = {
    0: "not processed", 1: "bowtie/trim", 2: "sun glint/obs.",
    3: "water", 4: "cloud", 5: "clear land", 6: "unknown",
    7: "low conf fire", 8: "nom conf fire", 9: "high conf fire",
}
FIRE_CLASS_COLORS = [
    "#d3d3d3",  # 0  grey
    "#a9a9a9",  # 1  dark grey
    "#b0c4de",  # 2  steel blue
    "#4682b4",  # 3  blue
    "#ffffff",  # 4  white
    "#90ee90",  # 5  light green
    "#ffd700",  # 6  gold
    "#ff8c00",  # 7  dark orange
    "#ff4500",  # 8  orange-red
    "#8b0000",  # 9  dark red
]
FIRE_CMAP = mcolors.ListedColormap(FIRE_CLASS_COLORS)
FIRE_NORM = mcolors.BoundaryNorm(boundaries=list(range(11)), ncolors=10)


# ===========================================================================
# 1.  READ VNP14A1 HDF-EOS5
# ===========================================================================

def tile_extent(h, v):
    """
    Return the sinusoidal (x_min, y_max) upper-left corner of tile hHHvVV.
    x_min = SINU_X_ORIGIN + h * TILE_SIZE_M
    y_max = SINU_Y_ORIGIN - v * TILE_SIZE_M
    """
    x_ul = SINU_X_ORIGIN + h * TILE_SIZE_M
    y_ul = SINU_Y_ORIGIN - v * TILE_SIZE_M
    return x_ul, y_ul


def read_vnp14a1(h5_path):
    """
    Read FireMask and MaxFRP from a VNP14A1 HDF-EOS5 file and return
    a dict with arrays and the rasterio transform / CRS.

    HDF-EOS5 path: HDFEOS/GRIDS/VNP14A1_Grid/Data Fields/{FireMask, MaxFRP, QA}
    Tile ID is parsed from the filename (e.g. h08v05).

    MaxFRP is stored as int32 with scale_factor 0.1 (MW).
    Returns MaxFRP in MW as float32 (NaN where <= 0 or fill).
    """
    h5_path = Path(h5_path)

    # Parse tile coords from filename: VNP14A1.AYYYYDDD.hHHvVV.002.*.h5
    import re
    m = re.search(r"\.h(\d{2})v(\d{2})\.", h5_path.name)
    if not m:
        raise ValueError(f"Cannot parse tile h/v from filename: {h5_path.name}")
    h_tile = int(m.group(1))
    v_tile = int(m.group(2))
    print(f"VNP14A1 tile: h{h_tile:02d}v{v_tile:02d}")

    with h5py.File(h5_path, "r") as hf:
        # Try both possible group names (V001 vs V002 differ slightly)
        grid_candidates = [
            "HDFEOS/GRIDS/VNP14A1_Grid/Data Fields",
            "HDFEOS/GRIDS/VNP_Grid_Daily_1km_Fire/Data Fields",
        ]
        data_group = None
        for gc in grid_candidates:
            if gc in hf:
                data_group = hf[gc]
                print(f"  HDF-EOS5 grid group: {gc}")
                break
        if data_group is None:
            # Walk and find any group containing FireMask
            found = []
            hf.visititems(lambda n, o: found.append(n)
                          if isinstance(o, h5py.Dataset) and "FireMask" in n else None)
            if not found:
                raise KeyError(
                    f"Cannot find FireMask in {h5_path.name}. "
                    f"Root groups: {list(hf.keys())}"
                )
            # found[0] is e.g. "HDFEOS/GRIDS/.../Data Fields/FireMask"
            data_group = hf["/".join(found[0].split("/")[:-1])]
            print(f"  Found via walk: {found[0]}")

        fire_mask = data_group["FireMask"][:].astype(np.uint8)

        # MaxFRP: int32, scale 0.1 → MW; fill value typically -28672
        max_frp_raw = data_group["MaxFRP"][:]
        scale = data_group["MaxFRP"].attrs.get("scale_factor", 0.1)
        fill  = data_group["MaxFRP"].attrs.get("_FillValue", -28672)
        max_frp = max_frp_raw.astype(np.float32)
        max_frp[max_frp_raw == fill] = np.nan
        max_frp[max_frp_raw <= 0]   = np.nan
        max_frp *= float(scale)

        qa = data_group["QA"][:].astype(np.uint8) if "QA" in data_group else None

    nrows, ncols = fire_mask.shape
    x_ul, y_ul = tile_extent(h_tile, v_tile)
    pixel_size = TILE_SIZE_M / ncols   # 926.6 m for 1200×1200
    transform  = from_origin(x_ul, y_ul, pixel_size, pixel_size)

    print(f"  Shape: {fire_mask.shape}  pixel: {pixel_size:.1f} m")
    print(f"  Upper-left (sinu m): ({x_ul:.0f}, {y_ul:.0f})")
    print(f"  Fire pixels (class 7-9): {np.sum(fire_mask >= 7)}")

    return {
        "fire_mask": fire_mask,
        "max_frp":   max_frp,
        "qa":        qa,
        "transform": transform,
        "crs":       SINU_CRS,
        "pixel_size": pixel_size,
        "tile": (h_tile, v_tile),
    }


# ===========================================================================
# 2.  REPROJECT VNP14A1 TO MATCH VNP14IMG GRID
# ===========================================================================

def reproject_l3_to_img(l3, img_profile):
    """
    Reproject VNP14A1 arrays to exactly match the VNP14IMG GeoTIFF grid.

    Uses nearest-neighbour for FireMask (categorical) and bilinear for MaxFRP.

    Parameters
    ----------
    l3          : dict from read_vnp14a1()
    img_profile : rasterio profile of the 375m GeoTIFF

    Returns
    -------
    dict with 'fire_mask' and 'max_frp' at 375m resolution
    """
    dst_crs       = img_profile["crs"]
    dst_transform = img_profile["transform"]
    dst_height    = img_profile["height"]
    dst_width     = img_profile["width"]

    results = {}
    for key, arr, resampling in [
        ("fire_mask", l3["fire_mask"], Resampling.nearest),
        ("max_frp",   l3["max_frp"],   Resampling.bilinear),
    ]:
        src_dtype = arr.dtype
        dst_arr   = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=arr.astype(np.float32),
            destination=dst_arr,
            src_transform=l3["transform"],
            src_crs=l3["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        if key == "fire_mask":
            # Round back to integer; NaN (outside tile) → 255 sentinel
            out = np.where(np.isnan(dst_arr), 255, np.round(dst_arr)).astype(np.uint8)
        else:
            out = dst_arr

        results[key] = out

    return results


# ===========================================================================
# 3.  READ VNP14IMG GEOTIFF
# ===========================================================================

def read_img_geotiff(tif_path):
    """
    Read the 5-band GeoTIFF produced by vnp14img_to_geotiff.py.

    Bands:  1=fire_mask  2=algorithm_qa  3=confidence  4=frp  5=bt_i4
    NaN = fill / no-data (integer bands stored as float32 with NaN for nodata).
    """
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        fire_mask_f = src.read(1)   # float32, NaN = fill
        # frp is band 4
        frp_f = src.read(4) if src.count >= 4 else None

    # Convert fire_mask back to uint8 (NaN → 255 sentinel)
    fire_mask = np.where(np.isnan(fire_mask_f), 255, np.round(fire_mask_f)).astype(np.uint8)
    frp = frp_f  # already float32 / NaN

    print(f"\nVNP14IMG GeoTIFF: {tif_path.name}")
    print(f"  Shape: {fire_mask.shape}  pixel: ~375 m")
    print(f"  CRS: {profile['crs'].to_string()[:60]}")
    print(f"  Fire pixels (class 7-9): {np.sum((fire_mask >= 7) & (fire_mask <= 9))}")

    return fire_mask, frp, profile


# ===========================================================================
# 4.  COMPARISON STATISTICS
# ===========================================================================

def fire_agreement(img_mask, l3_mask):
    """
    Compute per-class and overall agreement between two fire mask arrays.
    Both should be uint8 with 0-9 valid classes; 255 = no-data.

    Returns a dict of statistics.
    """
    valid = (img_mask != 255) & (l3_mask != 255)

    img_v = img_mask[valid]
    l3_v  = l3_mask[valid]

    n_total   = int(valid.sum())
    n_agree   = int((img_v == l3_v).sum())
    pct_agree = 100.0 * n_agree / n_total if n_total else 0.0

    # Fire / non-fire binary agreement
    img_fire = img_v >= 7
    l3_fire  = l3_v  >= 7
    tp = int(( img_fire &  l3_fire).sum())
    fp = int(( img_fire & ~l3_fire).sum())
    fn = int((~img_fire &  l3_fire).sum())
    tn = int((~img_fire & ~l3_fire).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    # Per-class pixel counts
    per_class = {}
    for c in range(10):
        per_class[c] = {
            "img": int((img_v == c).sum()),
            "l3":  int((l3_v  == c).sum()),
        }

    return {
        "n_valid_pixels": n_total,
        "pct_exact_agree": pct_agree,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "per_class": per_class,
    }


def frp_stats(img_frp, l3_frp):
    """
    Compare FRP arrays where both have valid (non-NaN) fire detections.
    Returns dict of summary stats.
    """
    from scipy.stats import pearsonr, spearmanr

    valid = np.isfinite(img_frp) & np.isfinite(l3_frp) & (img_frp > 0) & (l3_frp > 0)
    n = int(valid.sum())
    if n < 3:
        return {"n_colocated": n, "note": "fewer than 3 co-located fire pixels"}

    x = img_frp[valid]
    y = l3_frp[valid]
    bias    = float(np.mean(x - y))
    rmse    = float(np.sqrt(np.mean((x - y) ** 2)))
    r, _    = pearsonr(x, y)
    rho, _  = spearmanr(x, y)

    return {
        "n_colocated": n,
        "bias_MW":     bias,
        "rmse_MW":     rmse,
        "pearson_r":   float(r),
        "spearman_rho": float(rho),
        "img_mean_MW": float(x.mean()),
        "l3_mean_MW":  float(y.mean()),
    }


# ===========================================================================
# 5.  PLOT
# ===========================================================================

def make_figure(img_mask, l3_mask_repr, img_frp, l3_frp_repr, stats, out_path):
    """
    2-row × 3-column comparison figure.
      [0,0] VNP14IMG 375m fire mask
      [0,1] VNP14A1 1km fire mask (reprojected to 375m grid)
      [0,2] Difference map (img class - l3 class; 0=agree)
      [1,0] Fire pixel agreement bar chart (per class)
      [1,1] FRP scatter (img vs l3) at co-located fire pixels
      [1,2] Text summary of key statistics
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "VNP14IMG 375m  vs  VNP14A1 1km  —  Fire Mask & FRP Comparison",
        fontsize=13, fontweight="bold", y=0.98,
    )

    valid_mask = (img_mask != 255) & (l3_mask_repr != 255)
    img_plot   = np.where(img_mask       == 255, np.nan, img_mask.astype(float))
    l3_plot    = np.where(l3_mask_repr   == 255, np.nan, l3_mask_repr.astype(float))

    # ---- [0,0] VNP14IMG fire mask ----
    ax = axes[0, 0]
    im = ax.imshow(img_plot, cmap=FIRE_CMAP, norm=FIRE_NORM,
                   interpolation="none", aspect="equal")
    ax.set_title("VNP14IMG  (375 m swath → sinusoidal)", fontsize=10)
    ax.axis("off")

    # ---- [0,1] VNP14A1 reprojected ----
    ax = axes[0, 1]
    ax.imshow(l3_plot, cmap=FIRE_CMAP, norm=FIRE_NORM,
              interpolation="none", aspect="equal")
    ax.set_title("VNP14A1  (1 km L3 tile, reprojected to 375 m)", fontsize=10)
    ax.axis("off")

    # Shared colorbar for the two masks
    cbar = fig.colorbar(im, ax=axes[0, :2].ravel().tolist(),
                        orientation="horizontal", fraction=0.03, pad=0.05,
                        ticks=np.arange(0.5, 10))
    cbar.set_ticklabels([f"{c}: {FIRE_CLASSES[c]}" for c in range(10)],
                        fontsize=7)
    cbar.ax.tick_params(rotation=30)

    # ---- [0,2] Difference map ----
    ax = axes[0, 2]
    diff = np.where(valid_mask,
                    img_mask.astype(int) - l3_mask_repr.astype(int),
                    np.nan).astype(float)
    vmax = max(1, np.nanmax(np.abs(diff)))
    dm = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="none", aspect="equal")
    ax.set_title("Difference  (IMG class − L3 class)", fontsize=10)
    ax.axis("off")
    fig.colorbar(dm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)

    # ---- [1,0] Per-class pixel count bar chart ----
    ax = axes[1, 0]
    classes = list(range(10))
    img_counts = [stats["fire"]["per_class"][c]["img"] for c in classes]
    l3_counts  = [stats["fire"]["per_class"][c]["l3"]  for c in classes]
    x = np.arange(len(classes))
    w = 0.38
    ax.bar(x - w/2, img_counts, width=w, label="VNP14IMG", color="#e07b39", alpha=0.8)
    ax.bar(x + w/2, l3_counts,  width=w, label="VNP14A1",  color="#4a90d9", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], fontsize=8)
    ax.set_xlabel("Fire mask class", fontsize=9)
    ax.set_ylabel("Pixel count (valid overlap)", fontsize=9)
    ax.set_title("Per-class pixel distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=10)

    # ---- [1,1] FRP scatter ----
    ax = axes[1, 1]
    fs = stats.get("frp", {})
    if fs.get("n_colocated", 0) >= 3:
        valid_frp = np.isfinite(img_frp) & np.isfinite(l3_frp_repr) & (img_frp > 0) & (l3_frp_repr > 0)
        x_frp = img_frp[valid_frp]
        y_frp = l3_frp_repr[valid_frp]
        ax.scatter(x_frp, y_frp, s=6, alpha=0.4, color="#e07b39", rasterized=True)
        lim = max(x_frp.max(), y_frp.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
        ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
        ax.set_xlabel("VNP14IMG FRP (MW)", fontsize=9)
        ax.set_ylabel("VNP14A1 MaxFRP (MW)", fontsize=9)
        ax.set_title(
            f"FRP comparison  (n={fs['n_colocated']:,})\n"
            f"r={fs['pearson_r']:.3f}  bias={fs['bias_MW']:+.1f} MW  RMSE={fs['rmse_MW']:.1f} MW",
            fontsize=9,
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"Insufficient co-located fire pixels\n(n={fs.get('n_colocated',0)})",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("FRP comparison", fontsize=10)
    ax.tick_params(labelsize=8)

    # ---- [1,2] Summary text ----
    ax = axes[1, 2]
    ax.axis("off")
    fs2 = stats["fire"]
    lines = [
        "FIRE MASK AGREEMENT",
        f"  Valid pixels (overlap):   {fs2['n_valid_pixels']:,}",
        f"  Exact class match:        {fs2['pct_exact_agree']:.1f} %",
        "",
        "BINARY FIRE/NON-FIRE  (class ≥ 7 = fire)",
        f"  True positives (both):    {fs2['tp']:,}",
        f"  False positives (IMG only):{fs2['fp']:,}",
        f"  False negatives (L3 only): {fs2['fn']:,}",
        f"  True negatives:            {fs2['tn']:,}",
        f"  Precision:  {fs2['precision']:.3f}",
        f"  Recall:     {fs2['recall']:.3f}",
        f"  F1 score:   {fs2['f1']:.3f}",
    ]
    if stats.get("frp", {}).get("n_colocated", 0) >= 3:
        f = stats["frp"]
        lines += [
            "",
            "FRP  (co-located fire pixels)",
            f"  n co-located:   {f['n_colocated']:,}",
            f"  IMG mean FRP:   {f['img_mean_MW']:.1f} MW",
            f"  L3 mean FRP:    {f['l3_mean_MW']:.1f} MW",
            f"  Bias (IMG-L3):  {f['bias_MW']:+.1f} MW",
            f"  RMSE:           {f['rmse_MW']:.1f} MW",
            f"  Pearson r:      {f['pearson_r']:.3f}",
            f"  Spearman ρ:     {f['spearman_rho']:.3f}",
        ]
    ax.text(0.04, 0.96, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Summary statistics", fontsize=10)

    # Legend patches for fire classes
    patches = [mpatches.Patch(facecolor=FIRE_CLASS_COLORS[c],
                               edgecolor="grey", linewidth=0.5,
                               label=f"{c}: {FIRE_CLASSES[c]}")
               for c in range(10)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.01),
               title="Fire mask classes (shared by both products)")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    return fig


# ===========================================================================
# 6.  OPTIONAL: SAVE REPROJECTED VNP14A1 AS GEOTIFF
# ===========================================================================

def save_reprojected_l3(l3_repr, img_profile, out_path):
    """
    Save the reprojected VNP14A1 arrays as a 2-band GeoTIFF:
      Band 1: FireMask (uint8 stored as float32, NaN=nodata)
      Band 2: MaxFRP   (float32, MW, NaN=nodata)
    """
    p = img_profile.copy()
    p.update(count=2, dtype="float32", nodata=np.nan,
             compress="lzw", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(out_path, "w", **p) as dst:
        fm = l3_repr["fire_mask"].astype(np.float32)
        fm[fm == 255] = np.nan
        dst.write(fm, 1)
        dst.write(l3_repr["max_frp"], 2)
        dst.update_tags(
            PRODUCT="VNP14A1.002",
            NOTE="Reprojected from native 1km sinusoidal tile to match VNP14IMG 375m grid",
            BAND1="FireMask (0-9 classes)",
            BAND2="MaxFRP (MW)",
        )
    print(f"Reprojected VNP14A1 saved: {out_path}")


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare VNP14IMG 375m GeoTIFF vs VNP14A1 1km HDF-EOS5 tile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--img",  required=True, metavar="PATH",
                        help="VNP14IMG-derived 375m sinusoidal GeoTIFF")
    parser.add_argument("--l3",   required=True, metavar="PATH",
                        help="VNP14A1 HDF-EOS5 tile (.h5)")
    parser.add_argument("--save-l3-tif", action="store_true",
                        help="Also write the reprojected VNP14A1 as a GeoTIFF")
    parser.add_argument("--outdir", metavar="DIR",
                        help="Output directory (default: same directory as --img)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default 150)")
    args = parser.parse_args()

    img_path = Path(args.img)
    l3_path  = Path(args.l3)
    out_dir  = Path(args.outdir) if args.outdir else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists(): sys.exit(f"IMG file not found: {img_path}")
    if not l3_path.exists():  sys.exit(f"L3 file not found: {l3_path}")

    # ------------------------------------------------------------------
    # Read both products
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Reading VNP14IMG GeoTIFF …")
    img_mask, img_frp, img_profile = read_img_geotiff(img_path)

    print("\nReading VNP14A1 HDF-EOS5 tile …")
    l3 = read_vnp14a1(l3_path)

    # ------------------------------------------------------------------
    # Reproject VNP14A1 to match VNP14IMG grid
    # ------------------------------------------------------------------
    print("\nReprojecting VNP14A1 to 375m VNP14IMG grid …")
    l3_repr = reproject_l3_to_img(l3, img_profile)

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    print("\nComputing statistics …")
    fire_stats = fire_agreement(img_mask, l3_repr["fire_mask"])
    frp_stats_ = frp_stats(img_frp, l3_repr["max_frp"]) if img_frp is not None else {}

    stats = {"fire": fire_stats, "frp": frp_stats_}

    # Print to terminal
    print("\n--- FIRE MASK ---")
    print(f"  Valid overlap pixels : {fire_stats['n_valid_pixels']:,}")
    print(f"  Exact class match    : {fire_stats['pct_exact_agree']:.1f} %")
    print(f"  TP / FP / FN / TN    : "
          f"{fire_stats['tp']:,} / {fire_stats['fp']:,} / "
          f"{fire_stats['fn']:,} / {fire_stats['tn']:,}")
    print(f"  Precision / Recall / F1 : "
          f"{fire_stats['precision']:.3f} / {fire_stats['recall']:.3f} / {fire_stats['f1']:.3f}")

    if frp_stats_.get("n_colocated", 0) >= 3:
        f = frp_stats_
        print("\n--- FRP ---")
        print(f"  Co-located fire pixels : {f['n_colocated']:,}")
        print(f"  Bias (IMG - L3)        : {f['bias_MW']:+.1f} MW")
        print(f"  RMSE                   : {f['rmse_MW']:.1f} MW")
        print(f"  Pearson r              : {f['pearson_r']:.3f}")
    else:
        print(f"\n--- FRP: {frp_stats_.get('note', 'insufficient co-located pixels')} ---")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig_path = out_dir / f"comparison_{img_path.stem}_vs_{l3_path.stem}.png"
    make_figure(img_mask, l3_repr["fire_mask"], img_frp, l3_repr["max_frp"],
                stats, fig_path)

    # ------------------------------------------------------------------
    # Optionally save reprojected L3 GeoTIFF
    # ------------------------------------------------------------------
    if args.save_l3_tif:
        l3_tif = out_dir / f"{l3_path.stem}_reprojected_375m.tif"
        save_reprojected_l3(l3_repr, img_profile, l3_tif)

    print("\nDone.")


if __name__ == "__main__":
    main()
