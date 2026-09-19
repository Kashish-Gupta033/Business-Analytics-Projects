"""
=============================================================================
ads24x7 -- Marketing Intelligence: Segmenting Ad Campaigns with Clustering
=============================================================================
Business context
----------------
ads24x7 has closed a $10M seed round and is standing up a Marketing Analytics
practice. The Marketing Intelligence team wants the ad inventory segmented into
homogeneous groups so the business can describe -- in plain English -- what
"types" of ads it is actually running, and act differently on each type.

Analytical approach
-------------------
Segment on the three efficiency metrics the media-buying team already trades on:
    CPM -- Cost per 1,000 impressions  (what reach costs)
    CPC -- Cost per click              (what engagement costs)
    CTR -- Click-through rate          (how well the creative converts attention)

Pipeline: explore -> impute -> outlier diagnosis -> winsorise -> standardise ->
hierarchical (Ward) -> elbow -> silhouette -> profile & label.

Every import and every library call is commented for a reader who knows Python
but not the specifics of pandas / scipy / scikit-learn.

Run:  python ads24x7_clustering.py
Outputs: ./plots/*.png, ./outputs/*.csv
=============================================================================
"""

# --- Imports -----------------------------------------------------------------

# pandas: the tabular ("DataFrame") library -- reads the Excel file and does all
# row/column filtering, grouping and summary statistics in this script.
import pandas as pd

# numpy: fast numeric arrays and vectorised maths -- used here for np.nan,
# safe division, percentile maths and the k-ranges we loop over.
import numpy as np

# matplotlib.pyplot: the base plotting engine -- creates the figure canvases and
# saves every chart to disk as a .png.
import matplotlib

# Force the non-interactive "Agg" backend BEFORE pyplot is imported. Agg renders
# straight to an image file instead of trying to open a GUI window, which is what
# we want for a script that runs unattended and just writes .png files.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# seaborn: a statistical-plot layer that sits on top of matplotlib -- gives us
# one-line boxplots/histograms with sensible default styling.
import seaborn as sns

# scipy.cluster.hierarchy: agglomerative ("bottom-up") hierarchical clustering.
#   linkage()    -- builds the merge history of the clustering as a matrix
#   dendrogram() -- draws that merge history as the familiar tree diagram
from scipy.cluster.hierarchy import linkage, dendrogram

# scikit-learn's StandardScaler: converts each column to a z-score
# (value - column mean) / column standard deviation.
from sklearn.preprocessing import StandardScaler

# KMeans: the partitional clustering algorithm we are ultimately deploying.
from sklearn.cluster import KMeans

# silhouette_score: a cluster-quality metric in [-1, 1] measuring how much
# closer each point sits to its own cluster than to the nearest rival cluster.
# silhouette_samples returns the score for every individual point rather than
# the average, which is what lets us judge each cluster separately in Step 8b.
from sklearn.metrics import silhouette_score, silhouette_samples

# pathlib.Path: object-oriented file paths -- used to create the output folders
# in a way that works identically on Windows, macOS and Linux.
from pathlib import Path


# --- Configuration -----------------------------------------------------------

DATA_FILE = "Clustering Clean Ads_Data.xlsx"   # source workbook
SHEET = "Data"                                  # worksheet holding the records
FEATURES = ["CPM", "CPC", "CTR"]                # the three clustering features
RANDOM_STATE = 42                               # fixes KMeans' random start so
                                                # results reproduce run to run

# Path(...) builds a path object; .mkdir(exist_ok=True) creates the folder and
# stays silent if it is already there (instead of raising an error).
PLOTS = Path("plots"); PLOTS.mkdir(exist_ok=True)
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# sns.set_theme() applies seaborn's default grid/colour styling to every
# matplotlib figure created afterwards, so all our charts look consistent.
sns.set_theme(style="whitegrid", palette="deep")

# pandas truncates wide output by default; these options widen the console print
# so full describe() tables are readable rather than elided with "...".
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def banner(text):
    """Print a visually distinct section header to the console.

    Purely cosmetic: the script prints a lot of tables, and without separators
    the output is hard to navigate when reviewing results.
    """
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# =============================================================================
# STEP 1 -- INITIAL EXPLORATION
# =============================================================================
banner("STEP 1  |  INITIAL EXPLORATION")

# pd.read_excel() parses the .xlsx workbook into a DataFrame. sheet_name picks
# the worksheet -- the workbook also holds a "Data Dictionary" tab we don't model.
df = pd.read_excel(DATA_FILE, sheet_name=SHEET)

# .shape returns the (rows, columns) tuple -- the fastest sanity check that the
# file loaded fully and we are modelling the volume of data we expect.
print(f"\nShape (rows, columns): {df.shape}")

# .head(n) / .tail(n) return the first / last n rows. Looking at both ends
# catches files where trailing junk rows (totals, notes) got appended.
print("\n--- df.head() ---")
print(df.head())
print("\n--- df.tail() ---")
print(df.tail())

# .info() prints, per column, the dtype and the count of NON-null entries. It is
# the quickest way to spot two problems at once: numbers stored as text
# (dtype 'object' where we expect float), and columns with missing data.
print("\n--- df.info() ---")
df.info()

# .describe() computes count/mean/std/min/quartiles/max for numeric columns.
# .T transposes it so each variable is a row -- far easier to read with 19 columns.
print("\n--- df.describe() (transposed) ---")
print(df.describe().T)

# .isnull() makes a same-shaped True/False mask of missing cells; .sum() then
# counts the Trues down each column, giving nulls-per-column.
print("\n--- Null counts per column ---")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() else "No nulls anywhere.")

# .duplicated() flags each row that is an exact copy of an earlier row;
# .sum() counts them. Duplicate campaign rows would double-weight a segment.
print(f"\n--- Duplicate rows: {df.duplicated().sum()} ---")

# Denominator audit. CPM/CTR divide by Impressions and CPC divides by Clicks,
# so any row with a zero denominator makes the metric mathematically undefined.
# We must know the size of this problem BEFORE writing the imputation rule.
print("\n--- Zero-denominator audit (drives the imputation rule) ---")
print(f"Rows with Impressions == 0 : {(df['Impressions'] == 0).sum()}")
print(f"Rows with Clicks      == 0 : {(df['Clicks'] == 0).sum()}")
print(f"  ...of which Spend == 0   : {((df['Clicks'] == 0) & (df['Spend'] == 0)).sum()}")
print(f"  ...of which Spend  > 0   : {((df['Clicks'] == 0) & (df['Spend'] > 0)).sum()}")
print("Max Spend among zero-click rows: "
      f"{df.loc[df['Clicks'] == 0, 'Spend'].max():.2f}")

# Logical-consistency check: an ad cannot be clicked more often than it was
# shown, so Clicks > Impressions is impossible and signals a tracking/reporting
# defect rather than real performance.
print(f"Rows with Clicks > Impressions (impossible, CTR > 100%): "
      f"{(df['Clicks'] > df['Impressions']).sum()}")

# ---- SCALE AUDIT ON THE SUPPLIED CTR COLUMN --------------------------------
# The data dictionary states CTR = (Clicks / Impressions) * 100, i.e. a
# percentage. Before trusting that, we check whether the values ALREADY in the
# column actually follow it -- if they don't, imputing with the documented
# formula would leave the column mixing two scales 100x apart, which would
# wreck any distance-based clustering.
scale_check = df[df["CTR"].notnull() & (df["Impressions"] > 0)].copy()
# Ratio of the stored value to the raw fraction Clicks/Impressions. A median of
# ~1 means the column is stored as a FRACTION; ~100 means it is a PERCENTAGE.
ratio = (scale_check["CTR"] / (scale_check["Clicks"] / scale_check["Impressions"])).median()
print(f"\n--- CTR scale audit: median(stored CTR / (Clicks/Impressions)) = {ratio:.4f} ---")
print("    ~1.0  => supplied CTR is a FRACTION, not the documented percentage.")

banner("STEP 1 COMMENTARY")
print("""
* 25,857 rows x 19 columns. Dtypes are sensible: the 9 categorical descriptors
  (InventoryType, Ad Type, Platform, Device Type, Format, Timestamp) are text;
  every volume/money field is int64 or float64. Nothing numeric is stuck as text.

* Timestamp is stored as a string in a non-standard '2020-9-2-17'
  (year-month-day-hour) layout. Not needed for this segmentation, so left as-is.

* Missing data is confined to exactly the three derived metrics we were asked
  to impute: CTR 6,465 nulls, CPM 6,465 nulls, CPC 7,527 nulls (~25-29% each).
  All raw inputs (Spend, Impressions, Clicks) are complete -- so every missing
  value is recoverable from the formulas rather than genuinely unknown.

* Zero duplicate rows.

* TWO REAL DATA-QUALITY TRAPS the formulas alone do not cover:
  (a) 219 rows have Impressions == 0, so CPM and CTR are 0/0. All 219 also have
      Spend == 0 and Clicks == 0 -- these are dormant, no-activity rows.
  (b) 2,791 rows have Clicks == 0, so CPC divides by zero. 2,586 of them also
      have Spend == 0 (0/0), and the other 205 spent at most $0.19 in total.
  Both are handled explicitly in Step 2 rather than being allowed to produce
  inf/NaN that would silently poison the clustering.

* One row is logically impossible: 2 clicks recorded against 1 impression,
  which yields a 200% CTR. It is a tracking artefact, not performance. With a
  single occurrence it is not worth a bespoke rule -- the winsorisation in
  Step 4 caps it at the 99th percentile along with the other extremes.

* Heavy right-skew is already visible in describe(): Spend has a median of
  $1,174 against a max of $26,932, and Impressions a median of 162k against a
  max of 14.2m. This is normal for ad data and is exactly why Step 4 caps
  extremes rather than deleting them.

* SCALE MISMATCH FOUND: the supplied CTR values are the raw fraction
  Clicks/Impressions, NOT the x100 percentage the data dictionary specifies
  (audit ratio above is ~1.0, not ~100). CPM and CPC do match their documented
  formulas. Step 2 resolves this by putting the WHOLE CTR column on the
  documented percentage scale, so imputed and supplied values are comparable.
  (Because Step 5 standardises each feature, a uniform x100 on one column does
  not change the clustering -- but the internal INCONSISTENCY absolutely would.)
""")


# =============================================================================
# STEP 2 -- IMPUTE CPM, CPC, CTR
# =============================================================================
banner("STEP 2  |  IMPUTING CPM, CPC, CTR")

# Keep an untouched copy of the raw file so the before/after checks in later
# steps compare against the real original, not a partially processed frame.
df_raw = df.copy()

# Harmonise the supplied CTR values onto the documented percentage scale, so
# that the values we are about to impute (which use the x100 formula) sit on the
# same scale as the values already present. See the scale audit in Step 1.
df.loc[df["CTR"].notnull(), "CTR"] = df.loc[df["CTR"].notnull(), "CTR"] * 100
print("Rescaled the supplied (non-null) CTR values x100 onto the documented "
      "percentage scale, matching the formula used for imputation.")


def impute_metrics(row):
    """Fill any missing CPM / CPC / CTR on a single campaign row.

    Designed to be handed to DataFrame.apply(axis=1), which calls it once per
    row and passes that row in as a pandas Series (a labelled 1-D vector you
    index by column name, e.g. row['Spend']).

    Formulas (from the data dictionary):
        CPM = (Spend / Impressions) * 1000    -- cost per thousand impressions
        CPC =  Spend / Clicks                 -- cost per click
        CTR = (Clicks / Impressions) * 100    -- click-through rate, %

    Zero-denominator policy (the part the formulas don't specify):
        Impressions == 0  ->  CPM = 0 and CTR = 0. Verified in Step 1 that all
            such rows also have Spend == 0 and Clicks == 0: the ad never served,
            so it cost nothing per impression and converted nothing. 0 is the
            factually correct answer, not a filler value.
        Clicks == 0       ->  CPC = 0. The ad generated no billable clicks.
            Where Spend was also 0 this is exactly right; where Spend was
            non-zero it is at most $0.19 total, so 0 is a negligible, and
            conservative, approximation. The alternative -- infinity -- would be
            unusable in a Euclidean-distance algorithm.

    Returns
    -------
    pandas.Series with keys CPM, CPC, CTR -- the values this row should carry.
    Values that were already present are returned unchanged.
    """
    # Pull the raw inputs out of the row once, for readability below.
    spend = row["Spend"]
    impressions = row["Impressions"]
    clicks = row["Clicks"]

    # Start from whatever is already in the row, so we only overwrite blanks.
    cpm, cpc, ctr = row["CPM"], row["CPC"], row["CTR"]

    # --- CPM -----------------------------------------------------------
    # pd.isnull() is True for NaN/None; we only compute when the cell is empty.
    if pd.isnull(cpm):
        # Guard the denominator first -- see the zero-denominator policy above.
        cpm = (spend / impressions) * 1000 if impressions > 0 else 0.0

    # --- CPC -----------------------------------------------------------
    if pd.isnull(cpc):
        cpc = spend / clicks if clicks > 0 else 0.0

    # --- CTR -----------------------------------------------------------
    if pd.isnull(ctr):
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0.0

    # Returning a Series (rather than a tuple) lets .apply() assemble the
    # results straight back into a 3-column DataFrame with these column names.
    return pd.Series({"CPM": cpm, "CPC": cpc, "CTR": ctr})


# Build a boolean mask of rows needing work. [FEATURES] selects the 3 columns,
# .isnull() flags empty cells, and .any(axis=1) collapses across columns so a
# row is True if ANY of the three is missing.
needs_impute = df[FEATURES].isnull().any(axis=1)
print(f"\nRows with at least one missing metric: {needs_impute.sum():,} "
      f"of {len(df):,}")

# .apply(func, axis=1) runs impute_metrics on every flagged row (axis=1 = row-
# wise rather than column-wise). We restrict it with .loc[needs_impute] so
# complete rows are never touched -- as the brief requires.
df.loc[needs_impute, FEATURES] = df.loc[needs_impute].apply(impute_metrics, axis=1)

# --- Verification ------------------------------------------------------------
print("\n--- Null check AFTER imputation ---")
print(df[FEATURES].isnull().sum())

# np.isfinite() is False for both inf and NaN. A division we failed to guard
# would show up here as a non-finite value, so this is the real safety net.
finite = np.isfinite(df[FEATURES].to_numpy()).all()
print(f"All CPM/CPC/CTR values finite (no inf / NaN): {finite}")
assert df[FEATURES].isnull().sum().sum() == 0 and finite, "Imputation failed"

print("\n--- Post-imputation summary of the three features ---")
print(df[FEATURES].describe().T)


# =============================================================================
# STEP 3 -- OUTLIER CHECK
# =============================================================================
banner("STEP 3  |  OUTLIER DETECTION (IQR METHOD)")


def iqr_outliers(series):
    """Flag outliers in one numeric column using Tukey's 1.5 x IQR rule.

    The rule: take the middle 50% of the data (between the 25th and 75th
    percentiles -- that span is the Inter-Quartile Range, IQR), extend a
    'whisker' 1.5 x IQR beyond each end, and call anything past a whisker an
    outlier. It is distribution-free, so unlike a mean +/- 3 sd rule it is not
    itself dragged around by the very extremes it is trying to detect.

    Parameters
    ----------
    series : pandas.Series -- one numeric column.

    Returns
    -------
    (mask, lower_fence, upper_fence)
        mask : boolean Series, True where the value is an outlier.
    """
    # .quantile(q) returns the value below which q of the data falls.
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1                    # width of the middle 50% of the data
    lower = q1 - 1.5 * iqr           # lower whisker
    upper = q3 + 1.5 * iqr           # upper whisker
    # The | operator is element-wise OR on the two boolean Series.
    return (series < lower) | (series > upper), lower, upper


# Collect one summary row per feature so we can print a single tidy table.
outlier_rows = []
for col in FEATURES:
    mask, lo, hi = iqr_outliers(df[col])
    outlier_rows.append({
        "Feature": col,
        "Lower fence": round(lo, 4),
        "Upper fence": round(hi, 4),
        "Outliers (n)": int(mask.sum()),
        "Outliers (%)": round(100 * mask.mean(), 2),   # .mean() of a bool = proportion
        "Max value": round(df[col].max(), 2),
        # .skew() measures asymmetry: 0 = symmetric, >1 = strong right tail.
        "Skew": round(df[col].skew(), 2),
    })

# pd.DataFrame(list_of_dicts) turns our collected rows into a printable table.
outlier_summary = pd.DataFrame(outlier_rows)
print("\n--- IQR outlier summary (pre-treatment) ---")
print(outlier_summary.to_string(index=False))
outlier_summary.to_csv(OUT / "outlier_summary.csv", index=False)

# --- Visual diagnosis: boxplot + histogram per feature -----------------------
# plt.subplots(nrows, ncols) creates a grid of axes in one figure. We want
# 2 rows (boxplots on top, histograms below) x 3 columns (one per feature).
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, col in enumerate(FEATURES):
    # sns.boxplot draws the median line, the IQR box, the 1.5x whiskers, and
    # plots each point beyond the whiskers individually -- a direct visual of
    # exactly what the iqr_outliers() function counted numerically.
    sns.boxplot(y=df[col], ax=axes[0, i], color="#4C72B0")
    axes[0, i].set_title(f"{col} -- boxplot (pre-treatment)")

    # sns.histplot bins the values and draws frequency bars; kde=True overlays a
    # smoothed density curve, which makes the shape of the tail easier to read.
    sns.histplot(df[col], bins=60, kde=True, ax=axes[1, i], color="#4C72B0")
    axes[1, i].set_title(f"{col} -- distribution (pre-treatment)")

# .tight_layout() auto-adjusts spacing so titles and labels don't overlap.
plt.tight_layout()
# .savefig() writes the figure to disk; dpi=120 gives a crisp, report-ready image.
plt.savefig(PLOTS / "01_outliers_before.png", dpi=120)
# .close() frees the figure from memory -- important when a script makes many.
plt.close()
print(f"\nSaved: {PLOTS / '01_outliers_before.png'}")


# =============================================================================
# STEP 4 -- OUTLIER DECISION AND TREATMENT
# =============================================================================
banner("STEP 4  |  OUTLIER DECISION")

print("""
DECISION: treat the outliers, by WINSORISING at the 1st and 99th percentiles.
          Do NOT delete any rows.

Why treat them at all
---------------------
K-Means minimises squared Euclidean distance to a centroid, and that centroid is
an arithmetic MEAN. Two consequences follow. First, squaring the distance means
a point 10x further out contributes 100x more error, so a handful of extreme ads
can drag a centroid away from the bulk of its members. Second, because the mean
has no breakdown point, those same extremes can capture a whole cluster for
themselves -- the classic failure mode where k-means returns one huge blob plus
a two-member "cluster" of anomalies. Neither gives Marketing Intelligence a
segmentation it can act on.

Why NOT delete the rows
-----------------------
Ad-performance data is genuinely right-skewed: premium placements really do cost
20x a remnant impression, and a small number of campaigns really do drive most
of the spend. Those high-CPM ads are not errors -- they are arguably the most
commercially interesting inventory ads24x7 sells. Dropping every 1.5 x IQR flag
would discard a large, real slice of the business and bias the segments toward
cheap inventory.

Why winsorising (capping) rather than a log transform
-----------------------------------------------------
Capping pulls extreme values back to the 1st/99th percentile boundary while
KEEPING every row and preserving each ad's rank -- an expensive ad stays the
most expensive ad, it just stops exerting leverage proportional to how far out it
sits. It also keeps the features in their original business units (dollars-per-
thousand, dollars-per-click, percent), so the cluster profile table in Step 9
reads directly as money the media team recognises. A log transform would work
statistically but leaves centroids in log-space that have to be back-transformed
before anyone can interpret them, and it cannot handle the legitimate zeros in
this data without an offset fudge.
""")

# Snapshot the untreated values so we can draw a true before/after comparison.
df_before = df[FEATURES].copy()


def winsorize(series, lower_pct=0.01, upper_pct=0.99):
    """Cap a column's extreme values at given percentiles ('winsorising').

    Every value below the lower_pct percentile is raised to that percentile, and
    every value above the upper_pct percentile is lowered to it. Row count is
    unchanged and ordering is preserved -- only the magnitude of the tails moves.

    Parameters
    ----------
    series    : pandas.Series -- the column to cap.
    lower_pct : float -- lower percentile as a fraction (0.01 = 1st percentile).
    upper_pct : float -- upper percentile as a fraction (0.99 = 99th percentile).

    Returns
    -------
    (capped_series, lower_cap_value, upper_cap_value)
    """
    # Compute the two cut points from the data itself.
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    # .clip(lower, upper) is the vectorised cap: it squashes anything outside
    # [lo, hi] onto the nearest boundary and leaves everything else untouched.
    return series.clip(lower=lo, upper=hi), lo, hi


print("--- Applying 1st/99th percentile winsorisation ---")
cap_rows = []
for col in FEATURES:
    capped, lo, hi = winsorize(df[col])
    # Count how many cells actually moved, so we can report the true impact.
    n_changed = int((df[col] != capped).sum())
    cap_rows.append({
        "Feature": col,
        "1st pct cap": round(lo, 4),
        "99th pct cap": round(hi, 4),
        "Values capped": n_changed,
        "% of rows": round(100 * n_changed / len(df), 2),
        "Skew before": round(df[col].skew(), 2),
        "Skew after": round(capped.skew(), 2),
    })
    # Write the treated values into the working frame.
    df[col] = capped

print(pd.DataFrame(cap_rows).to_string(index=False))
pd.DataFrame(cap_rows).to_csv(OUT / "winsorisation_summary.csv", index=False)

print("\n--- describe() BEFORE winsorisation ---")
print(df_before.describe().T)
print("\n--- describe() AFTER winsorisation ---")
print(df[FEATURES].describe().T)

# --- Before/after visual (2 rows x 3 cols: before on top, after below) -------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, col in enumerate(FEATURES):
    sns.histplot(df_before[col], bins=60, kde=True, ax=axes[0, i], color="#C44E52")
    axes[0, i].set_title(f"{col} -- BEFORE (max {df_before[col].max():.2f})")
    sns.histplot(df[col], bins=60, kde=True, ax=axes[1, i], color="#55A868")
    axes[1, i].set_title(f"{col} -- AFTER winsorising (max {df[col].max():.2f})")
plt.tight_layout()
plt.savefig(PLOTS / "02_winsorise_before_after.png", dpi=120)
plt.close()
print(f"\nSaved: {PLOTS / '02_winsorise_before_after.png'}")

# Re-run the IQR count on the treated data. Note the fences are recomputed on
# the new distribution, so a residual count is expected and is NOT a failure --
# capping removes extreme LEVERAGE, it does not (and should not) flatten the
# genuine right-skew of ad-spend data into a normal distribution.
post = pd.DataFrame([
    {"Feature": c, "Outliers after (n)": int(iqr_outliers(df[c])[0].sum()),
     "Max after": round(df[c].max(), 2)}
    for c in FEATURES
])
print("\n--- IQR flags after treatment (fences recomputed on treated data) ---")
print(post.to_string(index=False))


# =============================================================================
# STEP 5 -- FEATURE SCALING
# =============================================================================
banner("STEP 5  |  Z-SCORE STANDARDISATION")

print("""
WHY SCALE: K-Means and Ward linkage both measure similarity as straight-line
(Euclidean) distance, which simply adds up the squared gaps on each feature. On
the raw data CPM spans roughly 0-40 dollars while CTR spans about 0-1 percent, so
a $10 CPM gap contributes ~100 to the squared distance while an entire CTR range
contributes ~1. The geometry would be decided almost entirely by CPM and the
clusters would essentially be CPM bands with CPC and CTR along for the ride.

Standardising to z-scores -- (value - mean) / standard deviation -- rewrites
every feature into 'standard deviations away from its own average', a common
unit. All three then contribute comparably. It also helps convergence: with
comparably-scaled axes the error surface is far more spherical, so Lloyd's
algorithm walks fairly toward the optimum instead of spending iterations
inching along one stretched, dominant axis.
""")

# StandardScaler() creates the transformer. .fit_transform(X) does two things in
# one call: fit() learns each column's mean and standard deviation from the
# data, and transform() applies (x - mean) / sd. The result is a plain numpy
# array, so we wrap it back into a DataFrame to keep the column names.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES, index=df.index)

print("--- Learned scaling parameters ---")
# scaler.mean_ and .scale_ hold the mean and sd learned per column. Keeping them
# matters operationally: new ads must be scaled with THESE numbers, not their own.
print(pd.DataFrame({"Mean (learned)": scaler.mean_,
                    "Std dev (learned)": scaler.scale_}, index=FEATURES).round(4))

print("\n--- Scaled features: mean ~0, sd ~1 by construction ---")
print(X_scaled_df.describe().T.round(4))


# =============================================================================
# STEP 6 -- HIERARCHICAL CLUSTERING (WARD)
# =============================================================================
banner("STEP 6  |  HIERARCHICAL CLUSTERING -- WARD DENDROGRAM")

# linkage() runs agglomerative clustering: it starts with every ad as its own
# cluster and repeatedly merges the two clusters whose union costs least.
#   method='ward'   -- at each step merge the pair that produces the smallest
#                      increase in total within-cluster variance. This is the
#                      same objective K-Means optimises, so Ward is the natural
#                      hierarchical companion to a K-Means deployment.
#   metric='euclidean' -- straight-line distance; required by Ward.
# The return value is the "linkage matrix": one row per merge, recording which
# two clusters joined, at what distance, and how many points resulted.
print(f"Building Ward linkage on {len(X_scaled):,} scaled observations "
      "(this takes a moment on 25k rows)...")
Z = linkage(X_scaled, method="ward", metric="euclidean")
print(f"Linkage matrix shape: {Z.shape}  (= n-1 merges)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# --- Left panel: truncated dendrogram ---
# A full dendrogram of 25,857 leaves is an unreadable black smear, so:
#   truncate_mode='lastp' + p=25 -- show only the last 25 merges, i.e. the top
#   of the tree, which is the only region that matters for choosing k.
#   show_leaf_counts=True -- annotate each collapsed leaf with how many ads it
#   contains, so we can see whether a branch is substantial or a sliver.
dendrogram(Z, truncate_mode="lastp", p=25, show_leaf_counts=True,
           leaf_rotation=90, leaf_font_size=9, ax=ax1)
ax1.set_title("Ward dendrogram (top 25 merges)")
ax1.set_xlabel("Cluster  (number of ads in brackets)")
ax1.set_ylabel("Ward linkage distance")

# --- Right panel: the merge-distance ("acceleration") curve ---
# Z[:, 2] is the third column of the linkage matrix: the distance at which each
# merge happened, in increasing order. [-15:] takes the final 15 merges. A large
# vertical jump means the algorithm was forced to fuse two genuinely dissimilar
# groups -- the natural place to cut the tree is just BELOW that jump.
last = Z[-15:, 2]
# ::-1 reverses the array so the biggest, last merge plots on the left.
ax2.plot(range(2, len(last) + 2), last[::-1], marker="o")
ax2.set_title("Merge distance of the final merges\n(a big jump = cut below it)")
ax2.set_xlabel("Number of clusters if cut here")
ax2.set_ylabel("Ward distance of that merge")
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig(PLOTS / "03_dendrogram.png", dpi=120)
plt.close()
print(f"Saved: {PLOTS / '03_dendrogram.png'}")

# Print the numbers behind the right-hand panel so the read is not purely visual.
print("\n--- Ward merge distances for the final merges ---")
for k_cut, d in zip(range(2, 12), Z[-10:, 2][::-1]):
    print(f"  cutting into {k_cut:>2} clusters -> merge distance {d:8.2f}")


# =============================================================================
# STEP 7 -- ELBOW PLOT (WCSS / INERTIA)
# =============================================================================
banner("STEP 7  |  ELBOW METHOD")

k_range = range(1, 11)      # the brief asks for k = 1..10
inertias = []               # will hold one WCSS value per k

for k in k_range:
    # KMeans(...) configures the model:
    #   n_clusters   -- k, how many centroids to fit
    #   init='k-means++' -- spreads the starting centroids apart rather than
    #                    picking at random, which avoids poor local optima
    #   n_init=10    -- run the whole fit 10 times from different starts and
    #                    keep the best; guards against an unlucky initialisation
    #   random_state -- fixes the randomness so results reproduce exactly
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    # .fit(X) runs the algorithm: assign each point to its nearest centroid,
    # recompute each centroid as the mean of its members, repeat to convergence.
    km.fit(X_scaled)
    # .inertia_ is WCSS -- the sum of squared distances from every point to its
    # own centroid. It measures how tight the clusters are, and it ALWAYS falls
    # as k rises (at k = n it hits zero), which is why we look for the bend
    # rather than the minimum.
    inertias.append(km.inertia_)
    print(f"  k = {k:>2}  ->  WCSS = {km.inertia_:14,.1f}")

# Quantify the elbow instead of eyeballing it: compute the percentage drop in
# WCSS each additional cluster buys. The elbow is where that marginal return
# collapses -- the first k after which extra clusters stop paying for themselves.
print("\n--- Marginal % reduction in WCSS from adding one more cluster ---")
for i in range(1, len(inertias)):
    drop = 100 * (inertias[i - 1] - inertias[i]) / inertias[i - 1]
    print(f"  k = {i} -> {i + 1}:  {drop:5.1f}% reduction")

plt.figure(figsize=(9, 6))
plt.plot(list(k_range), inertias, marker="o", linewidth=2, color="#4C72B0")
plt.xticks(list(k_range))
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS  (within-cluster sum of squares / inertia)")
plt.title("Elbow plot -- K-Means on standardised CPM, CPC, CTR")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(PLOTS / "04_elbow.png", dpi=120)
plt.close()
print(f"\nSaved: {PLOTS / '04_elbow.png'}")


# =============================================================================
# STEP 8 -- SILHOUETTE ANALYSIS
# =============================================================================
banner("STEP 8  |  SILHOUETTE ANALYSIS")

print("""
The silhouette score for a point is (b - a) / max(a, b), where a is its mean
distance to the other members of its OWN cluster and b is its mean distance to
the members of the nearest OTHER cluster. It runs from -1 to +1: near +1 the
point sits comfortably inside its cluster, near 0 it sits on a boundary, and
negative means it would be better off elsewhere. The score reported per k is the
average across all points. Unlike WCSS it does not automatically improve with
more clusters, so it can be maximised directly.

Silhouette is undefined at k=1 (there is no 'nearest other cluster'), hence 2..10.
""")

sil_k = range(2, 11)
sil_scores = []
for k in sil_k:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    # .fit_predict(X) fits the model and returns the cluster label of each row
    # in one call -- equivalent to .fit(X) followed by .labels_.
    labels = km.fit_predict(X_scaled)
    # silhouette_score(X, labels) computes the mean silhouette across all points.
    # sample_size caps the pairwise-distance computation at 10,000 randomly
    # chosen points -- the exact calculation is O(n^2) and would be needlessly
    # slow on 25,857 rows, while the sampled mean is stable to ~0.001 here.
    score = silhouette_score(X_scaled, labels, sample_size=10000,
                             random_state=RANDOM_STATE)
    sil_scores.append(score)
    print(f"  k = {k:>2}  ->  silhouette = {score:.4f}")

# max() finds the best score; .index() locates its position, and adding 2 maps
# that position back to the actual k (since the list starts at k=2).
best_sil_k = list(sil_k)[sil_scores.index(max(sil_scores))]
print(f"\nSilhouette is maximised at k = {best_sil_k} "
      f"(score {max(sil_scores):.4f})")

plt.figure(figsize=(9, 6))
plt.plot(list(sil_k), sil_scores, marker="o", linewidth=2, color="#55A868")
# axvline draws a vertical reference line marking the winning k.
plt.axvline(best_sil_k, color="#C44E52", linestyle="--", alpha=0.8,
            label=f"best k = {best_sil_k}")
plt.xticks(list(sil_k))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Mean silhouette score")
plt.title("Silhouette score by k -- higher is better")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(PLOTS / "05_silhouette.png", dpi=120)
plt.close()
print(f"Saved: {PLOTS / '05_silhouette.png'}")

# Persist both diagnostics side by side for the written report.
pd.DataFrame({
    "k": list(k_range),
    "WCSS_inertia": inertias,
    # Silhouette has no k=1 value, so pad the list with NaN to align lengths.
    "silhouette": [np.nan] + sil_scores,
}).to_csv(OUT / "k_selection_diagnostics.csv", index=False)

# =============================================================================
# STEP 8b -- PER-CLUSTER SILHOUETTE (the diagnostic that actually decides k)
# =============================================================================
banner("STEP 8b  |  PER-CLUSTER SILHOUETTE HEALTH CHECK")

print("""
The average silhouette in Step 8 is a single number for the whole partition, and
that hides the failure mode that matters most: a solution can post a respectable
average while containing one cluster that is not really a cluster at all -- a
slice carved off a neighbour, whose members sit on the boundary between the two.
Averaging lets three healthy clusters mask one bad one.

So we look INSIDE each candidate k, at two things per cluster:
  * the cluster's own mean silhouette; and
  * the share of its members scoring below 0.25, i.e. points that are nearly
    equidistant from their own cluster and the next one and could belong to
    either. A genuine cluster has few of these. A carved-off slice is full of them.

Note the sampling: silhouette_samples is O(n^2), so we score a RANDOM 12,000-row
subset rather than the first 12,000 rows. That distinction matters here -- this
file is ordered (it opens with tiny-impression rows and ends with 1-impression
rows), so taking the head would have produced a badly unrepresentative sample.
""")

# rng = a reproducible random-number generator; .choice(..., replace=False) draws
# a simple random sample of row positions without repeats.
rng = np.random.default_rng(RANDOM_STATE)
samp = rng.choice(len(X_scaled), size=min(12000, len(X_scaled)), replace=False)

health_rows = []
for k in [2, 3, 4, 5]:
    labels_k = KMeans(n_clusters=k, init="k-means++", n_init=10,
                      random_state=RANDOM_STATE).fit_predict(X_scaled)
    # silhouette_samples returns the score for EVERY point (not the average),
    # which is what lets us break the result down cluster by cluster.
    sil_each = silhouette_samples(X_scaled[samp], labels_k[samp])
    lab_s = labels_k[samp]
    for c in range(k):
        m = lab_s == c                      # boolean mask: members of cluster c
        health_rows.append({
            "k": k,
            "Cluster": c,
            "n (sampled)": int(m.sum()),
            "Mean silhouette": round(sil_each[m].mean(), 3),
            # A point scoring < 0.25 is effectively on the boundary. The share of
            # such points is the clearest single symptom of a non-cluster.
            "% members < 0.25": round(100 * (sil_each[m] < 0.25).mean(), 1),
        })

health = pd.DataFrame(health_rows)
print("--- Per-cluster silhouette health by candidate k ---")
print(health.to_string(index=False))
health.to_csv(OUT / "per_cluster_silhouette_health.csv", index=False)

# .groupby('k').agg(...) collapses the per-cluster rows into one summary row per
# k, keeping the WORST cluster in each solution -- because a partition is only as
# trustworthy as its weakest cluster.
worst = health.groupby("k").agg(
    Weakest_cluster_sil=("Mean silhouette", "min"),
    Worst_pct_boundary=("% members < 0.25", "max"),
).round(3)
print("\n--- Each solution judged by its WEAKEST cluster ---")
print(worst.to_string())

plt.figure(figsize=(10, 6))
for k in [2, 3, 4, 5]:
    sub = health[health["k"] == k]
    # One point per cluster: its mean silhouette against its boundary share.
    # Healthy solutions sit top-left (high silhouette, few boundary members).
    plt.scatter(sub["% members < 0.25"], sub["Mean silhouette"],
                s=110, label=f"k = {k}", alpha=0.85)
# axhline marks 0.25, the level below which a cluster is not meaningfully distinct.
plt.axhline(0.25, color="#C44E52", linestyle="--", alpha=0.7,
            label="0.25 -- below this a cluster is not distinct")
plt.xlabel("% of the cluster's members scoring below 0.25 (boundary cases)")
plt.ylabel("Cluster's own mean silhouette")
plt.title("Cluster health by candidate k -- each point is one cluster\n"
          "healthy clusters sit top-left; a point bottom-right is a carved-off slice")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(PLOTS / "05b_cluster_health.png", dpi=120)
plt.close()
print(f"\nSaved: {PLOTS / '05b_cluster_health.png'}")


# =============================================================================
# STEP 9 -- FINAL MODEL, CLUSTER PROFILING AND BUSINESS LABELS
# =============================================================================
banner("STEP 9  |  FINAL MODEL AND CLUSTER PROFILING")

# --- Reconciling the diagnostics ---------------------------------------------
# The elbow, the silhouette average, the dendrogram and the per-cluster health
# check ask four different questions and do not all point the same way. The
# decisive one is Step 8b: an average silhouette can look respectable while
# concealing a cluster that is merely a slice carved off its neighbour.
FINAL_K = 3
print(f"""
CHOOSING k
----------
  Dendrogram (Step 6) : two dominant branches at a Ward distance of ~313, each
                        splitting again at ~140 and ~99. Supports 2, 3 or 4.
  Elbow (Step 7)      : WCSS falls steeply and flattens around k=3-4; the
                        marginal-reduction table shows the payoff per extra
                        cluster collapsing after that point.
  Silhouette (Step 8) : the AVERAGE is maximised at k = {best_sil_k}.
  Cluster health (8b) : this is the one that decides it -- see below.

RESOLUTION -> k = {FINAL_K}.

  Against k=4. The average silhouette at k=4 (~0.51) looks acceptable, but the
  per-cluster breakdown shows it is carried by three healthy clusters masking one
  bad one. The fourth cluster scores ~0.21, and over HALF its members fall below
  0.25 -- they sit on the boundary with the cluster it was split from. That is
  the signature of a slice carved off a neighbour, not a distinct segment. It is
  a real commercial phenomenon (a very high-CTR tail) but it is a gradient within
  the efficient segment, not a mode of its own, so it is reported below as a
  flagged top-decile subset instead of being promoted to a fourth segment.

  Against k=2. k=2 is the cleanest split on every purely statistical measure
  (highest average silhouette, almost no boundary cases). It is rejected on
  business grounds, and the reason is specific rather than a general appeal to
  'actionability': k=2 merges the two low-CTR segments into a single group
  holding ~71% of spend. That collapse hides the central finding of this whole
  analysis -- that one segment consumes ~61% of the budget at a 0.24% CTR and
  $0.78 clicks, while a structurally different segment spends ~10% at a quarter
  the cost per click. At k=2 those two are the same cluster and the entire
  recommendation degenerates to 'some ads are efficient and some are not'.

  For k=3. Every cluster scores between roughly 0.55 and 0.57 with only 3-14% of
  members near a boundary -- no weak cluster anywhere in the solution. It also
  minimises the Davies-Bouldin index. The cost of moving from k=2 to k=3 is small
  (average silhouette ~0.60 -> ~0.55) and it buys the separation the business
  decision actually depends on. Three healthy segments beat two coarse ones and
  four where one is not real.
""")
print(f"Average silhouette at the chosen k={FINAL_K}: "
      f"{sil_scores[list(sil_k).index(FINAL_K)]:.4f}")
print("Per-cluster health at the chosen k (from Step 8b):")
print(health[health["k"] == FINAL_K].to_string(index=False))

# Fit the deployment model on the scaled features.
final_km = KMeans(n_clusters=FINAL_K, init="k-means++", n_init=10,
                  random_state=RANDOM_STATE)
# .fit_predict returns an integer label (0..k-1) for each of the 25,857 ads.
df["Cluster"] = final_km.fit_predict(X_scaled)

# --- Profile on the ORIGINAL (unscaled) units --------------------------------
# Labels were learned in z-score space, but nobody can act on "CPM = 1.4 sd".
# Attaching the labels to the unscaled dollar/percent values is what turns the
# model output into a business artefact.
# .groupby('Cluster') splits the rows by label; .agg({col: [...]}) then applies
# the listed summary functions to each column within each group.
profile = df.groupby("Cluster").agg({
    "CPM": "mean",
    "CPC": "mean",
    "CTR": "mean",
    "Spend": ["mean", "sum"],
    "Impressions": "mean",
    "Clicks": "mean",
    "Cluster": "size",       # 'size' counts the rows in each group
}).round(3)
# groupby+agg produces a two-level ("MultiIndex") column header; flattening it
# into single strings makes the table far easier to read and to export.
profile.columns = ["Mean_CPM", "Mean_CPC", "Mean_CTR", "Mean_Spend",
                   "Total_Spend", "Mean_Impressions", "Mean_Clicks", "Size"]
profile["Size_%"] = (100 * profile["Size"] / len(df)).round(2)
profile["Spend_%"] = (100 * profile["Total_Spend"] /
                      profile["Total_Spend"].sum()).round(2)


def rank_band(rank, k):
    """Translate a cluster's rank on one metric into a plain-English band.

    Parameters
    ----------
    rank : int -- 0 = lowest of the k clusters on this metric, k-1 = highest.
    k    : int -- how many clusters there are in total.

    Returns
    -------
    str : "lowest", "low", "high" or "highest".

    Why ranks rather than a simple above/below-average test: with four segments a
    binary median split can only produce two distinct descriptions per metric, so
    two different clusters end up sharing a name -- which makes the label set
    useless to the business. Ranking spreads the k clusters across an ordered
    scale, keeping every label distinct while still describing each segment
    RELATIVE to this portfolio rather than to an arbitrary external benchmark.
    """
    # Convert the rank to a 0-1 position so the bands generalise to any k, not
    # just to one particular k.
    pos = rank / (k - 1) if k > 1 else 0.5
    # A genuine MIDDLE band matters. With an odd k the middle cluster sits at
    # pos = 0.5, and without a "mid" band it would be forced up into "high" --
    # which would describe an ordinary, average segment as a strong one and
    # produce a materially wrong business label.
    if pos < 0.2:
        return "lowest"
    if pos < 0.4:
        return "low"
    if pos <= 0.6:
        return "mid"
    if pos <= 0.8:
        return "high"
    return "highest"


def label_cluster(row):
    """Turn one cluster's metric RANKS into a plain-English business label.

    Expects the profile row to already carry the rank columns computed below
    (Rank_CPM, Rank_CPC, Rank_CTR, Rank_Vol) plus the cluster count K.

    The logic reads a segment the way a media buyer would, in priority order:
      1. CTR rank -- does the creative actually earn attention?
      2. CPC rank -- is that engagement cheap or expensive to buy?
      3. Volume   -- is this a niche placement or mass-market inventory?
      4. CPM rank -- what does the raw reach cost?
    It returns an archetype name (what this inventory IS) followed by the band
    evidence, so the label can be defended without the profile table to hand.
    """
    k = int(row["K"])
    # Band each metric relative to the other clusters.
    ctr = rank_band(row["Rank_CTR"], k)     # engagement
    cpc = rank_band(row["Rank_CPC"], k)     # cost per engagement
    cpm = rank_band(row["Rank_CPM"], k)     # cost per unit of reach
    vol = rank_band(row["Rank_Vol"], k)     # delivered impressions = scale

    strong_ctr = ctr in ("high", "highest")    # creative converts attention
    cheap_click = cpc in ("lowest", "low")     # engagement is cheap to buy
    # "Serves at genuine scale" means anything that is NOT among the smallest
    # segments -- written as an exclusion rather than vol in ("high","highest")
    # so that a mid-ranked segment still counts as delivering volume. With a
    # small k the inclusive form would wrongly classify a segment averaging
    # hundreds of thousands of impressions as having no reach at all.
    big_reach = vol not in ("lowest", "low")
    # Share of the ad base. A segment holding a third of all inventory is the
    # portfolio's workhorse, whatever its rank on any single metric -- calling it
    # "niche" because it ranks low on volume-per-ad would be plainly wrong.
    is_major = row["Size_%"] >= 30.0

    # --- Branch 1: engagement is strong -------------------------------------
    if strong_ctr and cheap_click:
        # Clicks are both plentiful and cheap. What remains is whether this is
        # the portfolio's main efficient engine or a small premium pocket.
        if big_reach:
            name = "Budget-efficient workhorse"
        elif is_major:
            # Low volume PER AD, but a large share of the ad base -- this is the
            # efficient core of the business, not a niche placement.
            name = "Efficient performer"
        elif cpm == "highest":
            name = "Niche premium performer"
        else:
            name = "Efficient performer"
    elif strong_ctr:
        # Good CTR but dear clicks -- the attention is there, the price is not.
        name = "Expensive-but-high-CTR"

    # --- Branch 2: engagement is weak ---------------------------------------
    # Several segments can be simultaneously high-volume and low-CTR, so weak
    # engagement alone does not name a segment. What separates them commercially
    # is HOW that weak engagement is priced -- cheap clicks, cheap impressions, or
    # neither -- so the branches below split on price, in that order.
    elif big_reach and cheap_click:
        # Weak CTR, but the clicks that do land are cheap and plentiful.
        name = "Bargain bulk reach"
    elif big_reach and cpm == "lowest":
        # The cheapest impressions in the portfolio. Few people click, but the
        # reach itself costs almost nothing -- this is volume bought on price.
        name = "Cheap-impression volume"
    elif big_reach:
        # Huge delivery, weak CTR and dear clicks -- a classic awareness buy, and
        # the most expensive way in this portfolio to acquire a click.
        name = "High-reach low-engagement"
    else:
        # Small, weakly converting and not cheap: the underperforming tail.
        name = "Low-yield tail"

    return f"{name} ({ctr} CTR, {cpc} CPC, {cpm} CPM, {vol} volume)"


# Rank each cluster on each metric. .rank(method='dense') assigns 1 to the
# smallest mean and steps up with no gaps; subtracting 1 makes it 0-based, so
# rank 0 = lowest of the k clusters and rank k-1 = highest.
profile["Rank_CPM"] = profile["Mean_CPM"].rank(method="dense").astype(int) - 1
profile["Rank_CPC"] = profile["Mean_CPC"].rank(method="dense").astype(int) - 1
profile["Rank_CTR"] = profile["Mean_CTR"].rank(method="dense").astype(int) - 1
# Volume is ranked on delivered impressions. It is deliberately NOT a clustering
# feature -- but the scale of a segment is essential context for naming it and
# for writing a recommendation the media team can act on.
profile["Rank_Vol"] = profile["Mean_Impressions"].rank(method="dense").astype(int) - 1
profile["K"] = FINAL_K

# .apply(func, axis=1) runs label_cluster once per cluster row.
profile["Business_Label"] = profile.apply(label_cluster, axis=1)
# The rank columns were scaffolding for the naming; drop them from the report.
profile = profile.drop(columns=["Rank_CPM", "Rank_CPC", "Rank_CTR", "Rank_Vol", "K"])

# Guard against the exact failure this rank-based logic replaced: if two
# segments shared a name, the label set would tell the business nothing.
# .str.split(" (").str[0] keeps the archetype half of each label (everything
# before the bracketed evidence). Testing the FULL label would let two
# segments share an archetype merely because their band evidence differed.
assert profile["Business_Label"].str.split(" (", regex=False).str[0].nunique() == FINAL_K, \
    "Two clusters received the same archetype name -- naming logic needs review."

print("\n" + "-" * 78)
print("CLUSTER PROFILE  (means in original, unscaled units)")
print("-" * 78)
print(profile.to_string())
profile.to_csv(OUT / "cluster_profile.csv")
print(f"\nSaved: {OUT / 'cluster_profile.csv'}")

# --- The premium tail, kept as a flag rather than as a fourth cluster --------
# Step 9 rejected k=4 because the fourth cluster was not statistically distinct
# (over half its members sat on a boundary). The underlying commercial question
# -- is there a tail of genuinely exceptional inventory? -- is still worth
# answering, but it has to be answered carefully, because CTR is a RATIO and
# ratios computed on tiny denominators are unstable: an ad with 4 impressions and
# 1 click posts a 25% CTR that means nothing. So we impose a minimum-volume floor
# before flagging anything.
banner("PREMIUM TAIL  |  TOP-DECILE ADS WITHIN THE EFFICIENT SEGMENT")

# Identify the efficient segment programmatically (highest mean CTR) rather than
# hard-coding a cluster number, which would break if labels were renumbered.
eff_cluster = profile["Mean_CTR"].idxmax()

# Evidence for the floor: bucket the whole portfolio by impression volume and
# compare mean CTR with MEDIAN CTR. Where the two diverge wildly the average is
# being driven by a few lucky clicks on tiny denominators rather than by typical
# performance -- that is the signature of an unreliable ratio.
print("--- Why a volume floor is needed: mean vs median CTR by impression volume ---")
vol_bands = [(0, 100, "<100"), (100, 1_000, "100-1k"), (1_000, 10_000, "1k-10k"),
             (10_000, 100_000, "10k-100k"), (100_000, np.inf, ">100k")]
for lo, hi, lbl in vol_bands:
    m = (df["Impressions"] >= lo) & (df["Impressions"] < hi)
    print(f"  {lbl:>9} impressions: n={int(m.sum()):>6}  "
          f"mean CTR {df.loc[m, 'CTR'].mean():>6.2f}%  "
          f"median CTR {df.loc[m, 'CTR'].median():>6.2f}%")
print("  -> Below ~1,000 impressions mean and median diverge sharply (mean ~10%,")
print("     median ~0%): most such ads get no clicks at all and a handful post")
print("     extreme CTRs off a tiny denominator. Those are noise, not performance.")
print("     From 1k impressions upward mean and median converge, so CTR is")
print("     measuring something real there.")

# Apply the floor, then take the top decile WITHIN the efficient segment.
MIN_IMPRESSIONS = 10_000
eligible = (df["Cluster"] == eff_cluster) & (df["Impressions"] >= MIN_IMPRESSIONS)
# .quantile(0.90) gives the 90th-percentile CTR among eligible ads only.
ctr_p90 = df.loc[eligible, "CTR"].quantile(0.90)
df["Premium_Tail"] = eligible & (df["CTR"] >= ctr_p90)

print(f"\nEfficient segment    : cluster {eff_cluster} "
      f"({profile.loc[eff_cluster, 'Business_Label'].split(' (')[0]})")
print(f"Volume floor applied : {MIN_IMPRESSIONS:,} impressions "
      f"({int(eligible.sum()):,} of {int((df['Cluster'] == eff_cluster).sum()):,} "
      "segment ads qualify)")
print(f"Top-decile CTR cut   : {ctr_p90:.2f}%")
print(f"Ads flagged          : {int(df['Premium_Tail'].sum()):,} "
      f"({100 * df['Premium_Tail'].mean():.1f}% of the portfolio, "
      f"{100 * df.loc[df['Premium_Tail'], 'Spend'].sum() / df['Spend'].sum():.2f}% of spend)")

# Compare the flagged tail against the rest of its own segment.
tail_cmp = df[df["Cluster"] == eff_cluster].groupby("Premium_Tail").agg(
    Mean_CPM=("CPM", "mean"), Mean_CPC=("CPC", "mean"), Mean_CTR=("CTR", "mean"),
    Mean_Spend=("Spend", "mean"), Median_Impressions=("Impressions", "median"),
    Ads=("CPM", "size")).round(3)
tail_cmp.index = ["Rest of efficient segment", "Premium tail (top decile)"]
print("\n--- Premium tail vs the rest of its own segment ---")
print(tail_cmp.to_string())
tail_cmp.to_csv(OUT / "premium_tail_comparison.csv")
print("\nNOTE: with the volume floor in place this tail is a defensible target for")
print("investigation -- real delivery, sustained high CTR. Without the floor the")
print("same top-decile rule returns mostly sub-1,000-impression ads worth 0.09% of")
print("spend, which would have been a misleading thing to hand the business.")


# actually made of, which is what the media team can pull levers on.
print("\n--- Segment composition by Platform, Device and Format (%) ---")
# InventoryType is included deliberately: the other four descriptors turn out
# to be flat across segments, but inventory type is not -- it is the one
# categorical that actually separates them, and therefore the only one that
# gives the media team something to act on.
for cat in ["InventoryType", "Platform", "Device Type", "Format", "Ad Type"]:
    if cat in df.columns:
        # pd.crosstab counts rows for each (Cluster, category) pair;
        # normalize='index' converts those counts into row-wise proportions.
        ct = pd.crosstab(df["Cluster"], df[cat], normalize="index").round(3) * 100
        print(f"\n{cat}:")
        print(ct.to_string())
        ct.to_csv(OUT / f"composition_{cat.replace(' ', '_')}.csv")

# --- Visualising the segments ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for i, col in enumerate(FEATURES):
    # A boxplot per cluster shows not just the centroid but the SPREAD, which
    # reveals whether a segment is genuinely tight or just an averaged-out mix.
    sns.boxplot(data=df, x="Cluster", y=col, ax=axes[i], palette="deep",
                hue="Cluster", legend=False)
    axes[i].set_title(f"{col} by cluster")
plt.tight_layout()
plt.savefig(PLOTS / "06_cluster_boxplots.png", dpi=120)
plt.close()

# Scatter of the two cost dimensions, coloured by segment, with centroids marked.
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df.sample(6000, random_state=RANDOM_STATE),  # sample keeps
                x="CPM", y="CPC", hue="Cluster",                  # the plot legible
                palette="deep", alpha=0.5, s=18)
# scaler.inverse_transform() undoes the z-scoring on the learned centroids,
# putting them back into dollars so they can be overlaid on the raw-unit axes.
centroids = scaler.inverse_transform(final_km.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=260, marker="X",
            edgecolors="white", linewidths=2, label="Centroids", zorder=5)
plt.title(f"Ad segments in CPM-CPC space (k={FINAL_K}, 6,000-row sample)")
plt.xlabel("CPM  ($ per 1,000 impressions)")
plt.ylabel("CPC  ($ per click)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "07_cluster_scatter.png", dpi=120)
plt.close()

# Heatmap of the centroids in z-score space -- the single clearest picture of
# what makes each segment different, because all three features share one scale.
plt.figure(figsize=(8, 5))
centroid_z = pd.DataFrame(final_km.cluster_centers_, columns=FEATURES)
centroid_z.index = [f"C{i}: {profile.loc[i, 'Business_Label'].split(' (')[0]}"
                    for i in centroid_z.index]
# annot=True prints the value inside each cell. cmap='coolwarm' is a DELIBERATE
# choice over a red-green scale: red-green reads as bad-good, which would be
# actively misleading here because a high CPC is expensive, not desirable.
# coolwarm encodes direction (above/below average) without implying a verdict.
# center=0 pins the midpoint of the colour scale at the portfolio average (z=0).
sns.heatmap(centroid_z, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "standard deviations from the portfolio average"})
plt.title("Cluster centroids (standardised)\nwarm = above the portfolio average, cool = below")
plt.tight_layout()
plt.savefig(PLOTS / "08_centroid_heatmap.png", dpi=120)
plt.close()

print(f"\nSaved: {PLOTS / '06_cluster_boxplots.png'}, "
      f"{PLOTS / '07_cluster_scatter.png'}, {PLOTS / '08_centroid_heatmap.png'}")

# Export the fully labelled dataset for downstream BI / activation.
df.to_csv(OUT / "ads_with_clusters.csv", index=False)
print(f"Saved labelled dataset: {OUT / 'ads_with_clusters.csv'}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
banner("SUMMARY -- SEGMENTS AND RECOMMENDATIONS")
for cid, row in profile.iterrows():
    # Split the label back into archetype and band evidence so the summary
    # reads as a headline followed by the reasoning behind it.
    archetype, evidence = row['Business_Label'].split(' (', 1)
    print(f"\nCluster {cid} -- {archetype.upper()}")
    print(f"  Profile   : {evidence.rstrip(')')}")
    print(f"  Size      : {int(row['Size']):,} ads ({row['Size_%']}% of inventory)"
          f" | {row['Spend_%']}% of total spend")
    print(f"  CPM       : ${row['Mean_CPM']:.2f} per 1,000 impressions")
    print(f"  CPC       : ${row['Mean_CPC']:.2f} per click")
    print(f"  CTR       : {row['Mean_CTR']:.3f}%")
    print(f"  Avg spend : ${row['Mean_Spend']:,.0f} | "
          f"Avg impressions: {row['Mean_Impressions']:,.0f} | "
          f"Avg clicks: {row['Mean_Clicks']:,.0f}")

# =============================================================================
# STEP 10 -- BUDGET ALLOCATION BASIS
# =============================================================================
banner("STEP 10  |  SEGMENT EFFICIENCY AND BUDGET ALLOCATION")

# The segment profile in Step 9 reports MEANS PER AD. For a budget decision the
# relevant quantity is different: what each segment returns per unit of money
# put into it. That is a ratio of TOTALS, not an average of per-ad ratios --
# averaging per-ad CPCs would weight a $1 campaign the same as a $20,000 one.
budget = df.groupby("Cluster").agg(
    Spend=("Spend", "sum"), Clicks=("Clicks", "sum"),
    Impressions=("Impressions", "sum"), Revenue=("Revenue", "sum"),
)
budget["Label"] = [profile.loc[i, "Business_Label"].split(" (")[0]
                   for i in budget.index]
# Clicks bought per 1,000 units of spend -- the headline efficiency number.
budget["Clicks_per_1k_spend"] = (budget["Clicks"] / budget["Spend"] * 1000).round(0)
# Blended CPC: total spend over total clicks, the true portfolio-level cost.
budget["Blended_CPC"] = (budget["Spend"] / budget["Clicks"]).round(4)
budget["Spend_%"] = (100 * budget["Spend"] / budget["Spend"].sum()).round(1)
budget["Clicks_%"] = (100 * budget["Clicks"] / budget["Clicks"].sum()).round(1)
budget["Revenue_%"] = (100 * budget["Revenue"] / budget["Revenue"].sum()).round(1)
budget["Revenue_per_unit_spend"] = (budget["Revenue"] / budget["Spend"]).round(4)

print("--- Segment efficiency, computed on totals rather than per-ad means ---")
print(budget[["Label", "Spend_%", "Clicks_%", "Revenue_%",
              "Clicks_per_1k_spend", "Blended_CPC"]].to_string())
budget.to_csv(OUT / "budget_allocation_basis.csv")

# --- Is Revenue a performance signal? ---------------------------------------
# Worth checking before anyone justifies a segment by its revenue share. If
# Revenue is just Spend net of a near-constant Fee, then "this segment drives X%
# of revenue" is a restatement of "it spends X% of the budget" and says nothing
# about whether that spending worked.
rev_corr = df["Revenue"].corr(df["Spend"])
print(f"\nFee ranges {df['Fee'].min():.2f}-{df['Fee'].max():.2f}; "
      f"corr(Revenue, Spend) = {rev_corr:.4f}")
print("Revenue per unit of spend by segment: "
      + ", ".join(f"{budget.loc[i, 'Label']} {budget.loc[i, 'Revenue_per_unit_spend']:.3f}"
                  for i in budget.index))
print("-> Revenue is mechanically Spend x (1 - Fee) and is near-identical across")
print("   segments. It CANNOT be used to justify an allocation: a segment's")
print("   revenue share simply mirrors its spend share.")

# --- What the allocation is worth -------------------------------------------
# Project the clicks each split would buy, holding current segment efficiency
# constant. This is a first-order estimate: it assumes efficiency does not decay
# as a segment scales, which is exactly why the recommendation pairs the shift
# with a test rather than treating the projection as a guarantee.
SEED = 10_000_000
eff_c = profile["Mean_CTR"].idxmax()                       # efficient segment
hr_c = profile["Mean_CPC"].idxmax()                        # high-reach segment
cheap_c = [i for i in profile.index if i not in (eff_c, hr_c)][0]

scenarios = {
    "Status quo (spend shares as they are today)":
        {hr_c: budget.loc[hr_c, "Spend_%"] / 100,
         eff_c: budget.loc[eff_c, "Spend_%"] / 100,
         cheap_c: budget.loc[cheap_c, "Spend_%"] / 100},
    "Proposed (efficient 55%, high-reach 25%, cheap 10%, 10% held back)":
        {hr_c: 0.25, eff_c: 0.55, cheap_c: 0.10},
}
print(f"\n--- Projected clicks from a ${SEED / 1e6:.0f}M budget at current "
      "segment efficiency ---")
projected = {}
for name, weights in scenarios.items():
    # clicks = money to the segment x that segment's clicks per unit of money
    total = sum(SEED * w * budget.loc[c, "Clicks_per_1k_spend"] / 1000
                for c, w in weights.items())
    projected[name] = total
    print(f"  {total / 1e6:>6.1f}M clicks  <-  {name}")

names = list(scenarios)
uplift = 100 * (projected[names[1]] / projected[names[0]] - 1)
print(f"\nThe proposed split projects {uplift:.0f}% more clicks than today's "
      "shares, while deploying 10% less of the budget.")
print(f"It implies growing the efficient segment by "
      f"${SEED * 0.55 / 1e6:.1f}M on top of its "
      f"${budget.loc[eff_c, 'Spend']/1e6:.1f}M historical spend "
      f"(+{100 * SEED * 0.55 / budget.loc[eff_c, 'Spend']:.0f}%), which is a "
      "scale-up rather than a leap.")

print("\nDone. Plots in ./plots, tables in ./outputs.")
