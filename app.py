import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import core

st.set_page_config(page_title="AI Data Analysis Dashboard", page_icon="📊", layout="wide")
# ---------------------------------------------------------- theme ---------
# Cream background + Times New Roman. Applied here (not only in config.toml) so it
# also works on Streamlit Cloud and on computers set to dark mode.
BG, BG2, TEXT, ACCENT = "#FAF6EE", "#EFE7D6", "#2B2B2B", "#8B5E3C"
st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background-color: {BG}; }}
[data-testid="stSidebar"] {{ background-color: {BG2}; }}
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
.stApp li, .stApp button, .stApp input, .stApp textarea, .stApp span.st-emotion-cache,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-baseweb="select"] div, [data-baseweb="tab"], [data-testid="stCaptionContainer"] {{
    font-family: "Times New Roman", Times, serif !important;
}}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li {{ color: {TEXT}; }}
div[data-testid="stMetric"] {{ background:#FFFDF8; border:1px solid #DCCFB5;
    border-radius:10px; padding:12px 16px; }}
</style>
""", unsafe_allow_html=True)

# Plotly charts: same font, transparent background, warm colours
pio.templates["cream"] = go.layout.Template(layout=dict(
    font=dict(family="Times New Roman, Times, serif", color=TEXT),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#8B5E3C", "#2E6F8E", "#B8860B", "#5B7F4A", "#A0522D", "#6B5B95"]))
pio.templates.default = "plotly+cream"

ICON = {"warning": "⚠️", "anomaly": "🚨", "trend": "📈", "pattern": "🔎", "summary": "🧾"}

# ---------------------------------------------------------- sidebar --------
st.sidebar.title("AI Data Dashboard")
up = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx", "json"])
use_sample = st.sidebar.button("Load sample sales data")

if up is not None and st.session_state.get("name") != up.name:
    try:
        st.session_state.update(raw=core.load_file(up), name=up.name)
        st.session_state["clean"], st.session_state["log"] = core.clean(st.session_state["raw"])
    except ValueError as e:
        st.sidebar.error(str(e))
elif use_sample:
    st.session_state.update(raw=pd.read_csv("sample_sales.csv"), name="sample_sales.csv")
    st.session_state["clean"], st.session_state["log"] = core.clean(st.session_state["raw"])

if "raw" not in st.session_state:
    st.title("AI-Powered Data Analysis Dashboard")
    st.info("👈 Upload a CSV / Excel / JSON file, or load the sample sales data, to begin.")
    st.stop()

raw, df, name = st.session_state["raw"], st.session_state["clean"], st.session_state["name"]
num, cat, dt = core.split_cols(df)
page = st.sidebar.radio("Navigate", ["Overview", "Cleaning", "Visualize", "AI Insights",
                                     "Analytics", "Explorer", "Report"])
st.sidebar.caption(f"**{name}** · {len(df):,} rows × {df.shape[1]} cols (cleaned)")

# ---------------------------------------------------------- overview -------
if page == "Overview":
    st.title("Dataset overview")
    c = st.columns(5)
    c[0].metric("Rows", f"{len(raw):,}"); c[1].metric("Columns", raw.shape[1])
    c[2].metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
    c[4].metric("Numeric / Categorical", f"{len(num)} / {len(cat)}")
    schema = pd.DataFrame({"Type": raw.dtypes.astype(str), "Non-null": raw.notna().sum(),
                           "Missing": raw.isna().sum(), "Missing %": (raw.isna().mean() * 100).round(1),
                           "Unique": raw.nunique()})
    st.subheader("Columns & data types"); st.dataframe(schema, width="stretch")
    st.subheader("Statistical summary (raw data)")
    st.dataframe(raw.describe(include="all").T, width="stretch")
    st.subheader("Preview"); st.dataframe(raw.head(50), width="stretch")

# ---------------------------------------------------------- cleaning -------
elif page == "Cleaning":
    st.title("Data cleaning & preprocessing")
    c1, c2, c3 = st.columns(3)
    nf = c1.selectbox("Numeric missing values", ["median", "mean", "drop rows"])
    cf = c2.selectbox("Categorical missing values", ["mode", "Unknown"])
    th = c3.slider("Drop columns with missing above", 0.3, 1.0, 0.6, 0.05)
    dd = st.checkbox("Remove duplicate rows", True)
    sc = st.checkbox("Min-max normalise numeric columns", False)
    if st.button("Apply cleaning", type="primary"):
        st.session_state["clean"], st.session_state["log"] = core.clean(raw, nf, cf, dd, th, sc)
        st.rerun()
    st.subheader("Preprocessing summary")
    for line in st.session_state["log"]:
        st.write("✅", line)
    a, b = st.columns(2)
    a.metric("Missing cells: before → after", f"{int(raw.isna().sum().sum()):,} → {int(df.isna().sum().sum()):,}")
    b.metric("Rows: before → after", f"{len(raw):,} → {len(df):,}")
    st.download_button("Download cleaned CSV", df.to_csv(index=False), "cleaned_data.csv")

# ---------------------------------------------------------- visualize ------
elif page == "Visualize":
    st.title("Interactive visualisation")
    kind = st.selectbox("Chart type", ["Bar", "Line", "Pie", "Histogram", "Scatter", "Correlation heatmap"])
    allc = df.columns.tolist()
    if kind == "Bar":
        x = st.selectbox("Category (X)", cat + [c for c in allc if c not in cat])
        y = st.selectbox("Value (Y)", num); agg = st.selectbox("Aggregation", ["sum", "mean", "count", "max"])
        g = df.groupby(x)[y].agg(agg).reset_index().sort_values(y, ascending=False).head(30)
        fig = px.bar(g, x=x, y=y, title=f"{agg.title()} of {y} by {x}")
    elif kind == "Line":
        x = st.selectbox("X axis", dt + num + cat); ys = st.multiselect("Y axis", num, default=num[:1])
        d = df.sort_values(x)
        fig = px.line(d, x=x, y=ys, title="Line chart")
    elif kind == "Pie":
        n = st.selectbox("Category", cat); v = st.selectbox("Values", num)
        g = df.groupby(n)[v].sum().reset_index()
        fig = px.pie(g, names=n, values=v, title=f"{v} share by {n}")
    elif kind == "Histogram":
        x = st.selectbox("Column", num); bins = st.slider("Bins", 5, 100, 30)
        fig = px.histogram(df, x=x, nbins=bins, marginal="box", title=f"Distribution of {x}")
    elif kind == "Scatter":
        x = st.selectbox("X", num); y = st.selectbox("Y", num, index=min(1, len(num) - 1))
        col = st.selectbox("Colour by", [None] + cat)
        fig = px.scatter(df, x=x, y=y, color=col, opacity=.7, title=f"{y} vs {x}")
    else:
        fig = px.imshow(df[num].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Correlation heatmap")
    st.plotly_chart(fig, width="stretch", theme=None)

# ---------------------------------------------------------- AI insights ----
elif page == "AI Insights":
    st.title("AI-powered insights")
    insights = core.generate_insights(df)
    for i in insights:
        st.markdown(f"**{ICON.get(i['kind'], '•')} {i['title']}** — {i['text']}")
    st.subheader("Business recommendations")
    for r in core.recommendations(insights):
        st.write("👉", r)
    st.divider()
    st.subheader("AI narrative (optional LLM)")
    st.caption("Set the `ANTHROPIC_API_KEY` environment variable to enable. Only summary statistics are sent, never raw rows.")
    if st.button("Generate AI summary"):
        try:
            txt = core.ai_narrative(df, insights)
            st.session_state["narrative"] = txt
            st.write(txt) if txt else st.warning("No API key found - showing rule-based insights only.")
        except Exception as e:
            st.error(f"AI call failed: {e}")

# ---------------------------------------------------------- analytics ------
elif page == "Analytics":
    st.title("Dashboard analytics")
    if not num:
        st.warning("No numeric columns found."); st.stop()
    m = st.selectbox("Performance metric", num)
    c = st.columns(4)
    c[0].metric(f"Total {m}", f"{df[m].sum():,.0f}"); c[1].metric("Average", f"{df[m].mean():,.2f}")
    c[2].metric("Median", f"{df[m].median():,.2f}"); c[3].metric("Max", f"{df[m].max():,.2f}")

    if cat:
        k = st.selectbox("Compare by category", cat)
        g = df.groupby(k)[m].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        a, b = st.columns(2)
        a.subheader("Top performers"); a.plotly_chart(px.bar(g.head(5).reset_index(), x=k, y="sum"), width="stretch", theme=None)
        b.subheader("Comparison vs overall average")
        diff = ((g["mean"] / df[m].mean() - 1) * 100).rename("% vs avg").reset_index()
        b.plotly_chart(px.bar(diff, x=k, y="% vs avg", color="% vs avg", color_continuous_scale="RdYlGn"), width="stretch", theme=None)
    if dt:
        s = core.monthly(df, dt[0], m).to_frame(m)
        s["MoM growth %"] = s[m].pct_change() * 100
        st.subheader("Growth trend")
        a, b = st.columns(2)
        a.plotly_chart(px.line(s.reset_index(), x=s.index.name or "index", y=m, markers=True), width="stretch", theme=None)
        b.plotly_chart(px.bar(s.reset_index(), x=s.index.name or "index", y="MoM growth %"), width="stretch", theme=None)
    st.subheader("Distribution analysis")
    dist = df[num].agg(["mean", "median", "std", "skew", "kurt"]).T
    st.dataframe(dist.round(2), width="stretch")

# ---------------------------------------------------------- explorer -------
elif page == "Explorer":
    st.title("Search & filter")
    f = df.copy()
    q = st.text_input("🔎 Search any text column")
    if q:
        mask = pd.Series(False, index=f.index)
        for c in f.select_dtypes(["object", "category"]).columns:
            mask |= f[c].astype(str).str.contains(q, case=False, na=False)
        f = f[mask]
    with st.expander("Column filters", expanded=True):
        for c in st.multiselect("Filter columns", df.columns.tolist()):
            if c in num:
                lo, hi = float(df[c].min()), float(df[c].max())
                r = st.slider(c, lo, hi, (lo, hi)); f = f[f[c].between(*r)]
            elif c in dt:
                r = st.date_input(c, (df[c].min().date(), df[c].max().date()))
                if len(r) == 2:
                    f = f[(f[c] >= pd.Timestamp(r[0])) & (f[c] <= pd.Timestamp(r[1]))]
            else:
                sel = st.multiselect(f"{c} values", sorted(df[c].astype(str).unique()))
                if sel: f = f[f[c].astype(str).isin(sel)]
    cond = st.text_input("Custom condition (pandas query)", placeholder="e.g.  Revenue > 500 and Region == 'East'")
    if cond:
        try: f = f.query(cond)
        except Exception as e: st.error(f"Invalid condition: {e}")
    st.caption(f"{len(f):,} of {len(df):,} rows match")
    st.dataframe(f, width="stretch")
    st.download_button("Download filtered CSV", f.to_csv(index=False), "filtered.csv")

# ---------------------------------------------------------- report ---------
else:
    st.title("Report generation")
    st.write("Builds a self-contained HTML report: overview, charts, insights and recommendations.")
    if st.button("Generate report", type="primary"):
        ins = core.generate_insights(df)
        rep = core.build_report(df, name, ins, st.session_state.get("narrative"))
        st.download_button("⬇️ Download report (HTML)", rep, "analysis_report.html", "text/html")
        st.components.v1.html(rep, height=700, scrolling=True)
    st.caption("Tip: open the HTML in a browser and use Print → Save as PDF.")
