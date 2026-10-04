"""Loading/validation, cleaning, insight generation and reporting
for the AI Data Analysis Dashboard."""
import base64
import html
import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
import numpy as np
import pandas as pd
import seaborn as sns

ALLOWED_EXT = {".csv", ".xlsx", ".json"}
MAX_MB = 50


# ------------------------------------------------------------ loading ------
def load_file(f) -> pd.DataFrame:
    """Validate and read an uploaded file (Streamlit UploadedFile)."""
    ext = os.path.splitext(f.name.lower())[1]
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported file type '{ext}'. Use CSV, XLSX or JSON.")
    if f.size > MAX_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_MB} MB.")
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(f)
            except UnicodeDecodeError:
                f.seek(0)
                df = pd.read_csv(f, encoding="latin-1")
        elif ext == ".xlsx":
            df = pd.read_excel(f)
        else:
            df = pd.read_json(f)
    except Exception as e:
        raise ValueError(f"Could not parse file: {e}")
    if df.empty or df.shape[1] < 2:
        raise ValueError("The file is empty or has fewer than 2 columns.")
    return df


# ----------------------------------------------------------- cleaning ------
def clean(df, num_fill="median", cat_fill="mode", drop_dups=True,
          drop_col_thresh=0.6, scale=False):
    """Returns (clean_df, log[list of str])."""
    log = []
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    log.append(f"Stripped whitespace from column names; start shape {df.shape[0]:,} x {df.shape[1]}")

    if drop_dups:
        n = int(d.duplicated().sum())
        d = d.drop_duplicates()
        log.append(f"Removed {n:,} duplicate rows")

    sparse = [c for c in d.columns if d[c].isna().mean() > drop_col_thresh]
    if sparse:
        d = d.drop(columns=sparse)
        log.append(f"Dropped columns with >{drop_col_thresh:.0%} missing: {', '.join(sparse)}")

    # convert date-like text columns
    for c in d.select_dtypes("object").columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(d[c], errors="coerce")
        if d[c].notna().any() and parsed.notna().sum() / d[c].notna().sum() > 0.9:
            d[c] = parsed
            log.append(f"Converted '{c}' to datetime")
        else:  # numbers stored as text
            num = pd.to_numeric(d[c], errors="coerce")
            if d[c].notna().any() and num.notna().sum() / d[c].notna().sum() > 0.95:
                d[c] = num
                log.append(f"Converted '{c}' to numeric")

    for c in d.select_dtypes("number").columns:
        k = int(d[c].isna().sum())
        if k:
            if num_fill == "drop rows":
                d = d[d[c].notna()]
            else:
                d[c] = d[c].fillna(d[c].median() if num_fill == "median" else d[c].mean())
            log.append(f"Filled {k:,} missing in '{c}' ({num_fill})")
    for c in d.select_dtypes(["object", "category", "bool"]).columns:
        k = int(d[c].isna().sum())
        if k:
            fill = d[c].mode().iloc[0] if (cat_fill == "mode" and d[c].notna().any()) else "Unknown"
            d[c] = d[c].fillna(fill)
            log.append(f"Filled {k:,} missing in '{c}' with '{fill}'")
    for c in d.select_dtypes("datetime").columns:
        k = int(d[c].isna().sum())
        if k:
            d = d[d[c].notna()]
            log.append(f"Dropped {k:,} rows with invalid dates in '{c}'")

    if scale:
        nums = d.select_dtypes("number").columns
        rng = d[nums].max() - d[nums].min()
        d[nums] = (d[nums] - d[nums].min()) / rng.replace(0, 1)
        log.append(f"Min-max scaled {len(nums)} numeric columns")

    d = d.reset_index(drop=True)
    log.append(f"Final shape {d.shape[0]:,} x {d.shape[1]}")
    return d, log


# ----------------------------------------------------------- insights ------
def split_cols(df):
    num = df.select_dtypes("number").columns.tolist()
    dt = df.select_dtypes("datetime").columns.tolist()
    cat = [c for c in df.select_dtypes(["object", "category", "bool"]).columns
           if 1 < df[c].nunique() <= 50]
    return num, cat, dt


SUM_HINTS = ("revenue", "sales", "amount", "units", "quantity", "qty", "profit", "cost", "total", "count")


def default_agg(col):
    """Sum volume-like metrics, average rate/price-like ones."""
    n = str(col).lower()
    if any(w in n for w in ("discount", "rate", "price", "ratio", "percent", "%")):
        return "mean"
    return "sum" if any(h in n for h in SUM_HINTS) else "mean"


def monthly(df, date_col, val_col, agg=None):
    agg = agg or default_agg(val_col)
    g = df.groupby(df[date_col].dt.to_period("M"))[val_col].agg(agg)
    g.index = g.index.to_timestamp()
    return g


def generate_insights(df):
    """Rule-based insight engine. Returns list of {kind, title, text}."""
    out = []
    add = lambda kind, title, text: out.append({"kind": kind, "title": title, "text": text})
    num, cat, dt = split_cols(df)

    miss = df.isna().mean() * 100
    bad = miss[miss > 5]
    if len(bad):
        add("warning", "Data completeness",
            "Columns with >5% missing values: " + ", ".join(f"{c} ({p:.0f}%)" for c, p in bad.items()) + ".")

    for c in num[:15]:
        s = df[c].dropna()
        if s.nunique() < 10:
            continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        if iqr == 0:
            continue
        k = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if k / len(s) > 0.01:
            add("anomaly", f"Outliers in {c}",
                f"{k:,} values ({k / len(s):.1%}) fall outside the IQR fences "
                f"[{q1 - 1.5 * iqr:,.2f}, {q3 + 1.5 * iqr:,.2f}]; max is {s.max():,.2f}.")
        if abs(s.skew()) > 1.5:
            add("pattern", f"Skewed distribution: {c}",
                f"Skewness is {s.skew():.2f}, so the mean ({s.mean():,.2f}) differs a lot from the median "
                f"({s.median():,.2f}). Prefer the median for typical values.")

    if len(num) >= 2:
        corr = df[num].corr()
        seen = set()
        pairs = []
        for a in num:
            for b in num:
                if a < b and (a, b) not in seen and pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= 0.7:
                    pairs.append((a, b, corr.loc[a, b])); seen.add((a, b))
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2]))[:4]:
            add("pattern", f"Strong {'positive' if r > 0 else 'negative'} relationship",
                f"{a} and {b} have correlation {r:.2f}. They move {'together' if r > 0 else 'in opposite directions'}.")

    if dt and num:
        for m in num[:5]:
            s = monthly(df, dt[0], m)
            if len(s) >= 6 and s.iloc[:3].mean():
                change = (s.iloc[-3:].mean() / s.iloc[:3].mean() - 1) * 100
                word = "grown" if change > 0 else "declined"
                add("trend", f"Trend in {m}",
                    f"Monthly {'total' if default_agg(m) == 'sum' else 'average'} has {word} {abs(change):.1f}% (last 3 months vs first 3). "
                    f"Peak month: {s.idxmax():%b %Y}; lowest: {s.idxmin():%b %Y}.")

    if num:
        m = num[0]
        for c in cat[:3]:
            g = df.groupby(c)[m].sum().sort_values(ascending=False)
            if g.sum() > 0 and len(g) > 1:
                share = g.iloc[0] / g.sum()
                add("pattern", f"Top {c} by {m}",
                    f"'{g.index[0]}' leads with {share:.0%} of total {m}; lowest is '{g.index[-1]}' "
                    f"({g.iloc[-1] / g.sum():.0%}).")

    for c in cat[:5]:
        share = df[c].value_counts(normalize=True).iloc[0]
        if share > 0.6:
            add("warning", f"Imbalance in {c}",
                f"'{df[c].value_counts().index[0]}' makes up {share:.0%} of rows - analyses may be biased.")

    add("summary", "Dataset summary",
        f"{len(df):,} rows, {df.shape[1]} columns ({len(num)} numeric, {len(cat)} categorical, "
        f"{len(dt)} date). {int(df.isna().sum().sum()):,} missing cells remain.")
    return out


def recommendations(insights):
    kinds = {i["kind"] for i in insights}
    titles = " ".join(i["title"] for i in insights)
    recs = []
    if "anomaly" in kinds:
        recs.append("Investigate flagged outliers - confirm whether they are data-entry errors or genuine high-value events.")
    if "Imbalance" in titles:
        recs.append("Rebalance or segment analyses by the dominant category to avoid biased conclusions.")
    if any("declined" in i["text"] for i in insights if i["kind"] == "trend"):
        recs.append("Drill into declining metrics by category/region to find the source of the drop.")
    if any("grown" in i["text"] for i in insights if i["kind"] == "trend"):
        recs.append("Double down on drivers behind growing metrics; check whether growth is concentrated in few segments.")
    if any(i["title"].startswith("Top ") for i in insights):
        recs.append("Review dependence on the top-performing category and build plans to lift the weakest ones.")
    if "Strong" in titles:
        recs.append("Use strongly correlated variables as candidate drivers (or drop one of each redundant pair before modelling).")
    if "Data completeness" in titles:
        recs.append("Improve data collection for columns with heavy missingness.")
    return recs or ["No major issues found - consider collecting more dimensions for deeper analysis."]


def ai_narrative(df, insights):
    """Optional LLM summary (set ANTHROPIC_API_KEY). Only summary statistics are sent."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    stats = df.describe(include="all").T.head(25).to_string()
    facts = "\n".join(f"- {i['title']}: {i['text']}" for i in insights)
    prompt = ("You are a business data analyst. Using the summary statistics and detected findings below, "
              "write a concise executive summary (max 180 words) with 3 actionable recommendations.\n\n"
              f"SUMMARY STATS:\n{stats}\n\nFINDINGS:\n{facts}")
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=700,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


# ------------------------------------------------------------- report ------
def _png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_report(df, name, insights, narrative=None):
    """Self-contained HTML report (open in browser, Print -> Save as PDF)."""
    num, cat, dt = split_cols(df)
    imgs = []
    if len(num) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(df[num].corr(), annot=len(num) <= 10, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation heatmap"); imgs.append(_png(fig))
    for c in num[:2]:
        fig, ax = plt.subplots(figsize=(6, 3.5)); sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(f"Distribution of {c}"); imgs.append(_png(fig))
    if cat and num:
        g = df.groupby(cat[0])[num[0]].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5)); g.plot.bar(ax=ax, color="#4c72b0")
        ax.set_title(f"{num[0]} by {cat[0]} (top 10)"); imgs.append(_png(fig))
    if dt and num:
        s = monthly(df, dt[0], num[0])
        fig, ax = plt.subplots(figsize=(6, 3.5)); s.plot(ax=ax, marker="o")
        ax.set_title(f"Monthly {num[0]}"); imgs.append(_png(fig))

    e = html.escape
    li = "".join(f"<li><b>{e(i['title'])}:</b> {e(i['text'])}</li>" for i in insights)
    rec = "".join(f"<li>{e(r)}</li>" for r in recommendations(insights))
    schema = pd.DataFrame({"Type": df.dtypes.astype(str), "Missing": df.isna().sum(),
                           "Unique": df.nunique()}).reset_index().rename(columns={"index": "Column"})
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Report - {e(name)}</title>
<style>body{{font-family:"Times New Roman",Times,serif;max-width:900px;margin:30px auto;color:#222;line-height:1.5}}
h1{{border-bottom:3px solid #4c72b0}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}}th{{background:#f0f3f8}}
.grid{{display:flex;flex-wrap:wrap;gap:10px}}.grid img{{width:48%}}</style></head><body>
<h1>Data Analysis Report</h1><p><b>File:</b> {e(name)} &nbsp; <b>Rows:</b> {len(df):,} &nbsp; <b>Columns:</b> {df.shape[1]}</p>
<h2>1. Dataset overview</h2>{schema.to_html(index=False)}
<h2>2. Key visualisations</h2><div class="grid">{''.join(f'<img src="data:image/png;base64,{i}">' for i in imgs)}</div>
<h2>3. Insights</h2><ul>{li}</ul>
{f'<h2>AI summary</h2><p>{e(narrative)}</p>' if narrative else ''}
<h2>4. Recommendations</h2><ol>{rec}</ol></body></html>"""
