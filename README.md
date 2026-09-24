# Mobile Product Segmentation & Recommendation System

An end-to-end data analytics and machine learning pipeline that segments mobile
phone products using clustering, generates business insights per segment, and
powers a similarity-based recommendation system — delivered as an interactive
Streamlit web application.



---

## Project Files

| File | Description |
|---|---|
| `Mobile Reviews Sentiment null.csv` | Raw input dataset (50,000 reviews, 22 products) |
| `Global_mobile_review_prjt4.ipynb` | Full analysis notebook — cleaning, EDA, clustering, recommender |
| `product_segments_summary.csv` | Aggregated per-product output (brand, model, price, rating, segment) |
| `reviews_with_product_segments.csv` | Full review-level dataset joined with segment labels |
| `app.py` | Streamlit application (4 tabs — see below) |
| `PROJECT_README.md` | This file |

---

##  Pipeline Overview

### 1. Data Collection
- Loaded the raw CSV into a Pandas DataFrame
- Inspected shape, dtypes, and column structure

### 2. Data Preprocessing
- Handled missing values: group-median imputation for `price_usd`/`rating`,
  re-derived `price_local` from `price_usd × exchange_rate`, imputed `sentiment`
  from the numeric rating, filled `source` as `"Unknown"`
- Removed duplicate records
- Converted `review_date` to datetime and `price_local` to float
- Engineered `engagement_score` from `helpful_votes` and `verified_purchase`
- Encoded categorical variables (`brand`, `country`, `model`) via One-Hot Encoding
- Scaled numeric features with `StandardScaler`

### 3. Exploratory Data Analysis (EDA)
- Product distribution across brands and countries
- Top-rated and lowest-rated products
- Price vs. spec-rating correlations
- Review volume and rating trends over time
- Age-group and helpful-votes patterns

### 4. Clustering (Segmentation)
- Applied **K-Means** (k = 4) on average price and review volume
- Labeled clusters by price rank into business-friendly segments:
  **Budget → Mid-range → Upper-mid → Premium**
- Validated with silhouette score
- Exported `product_segments_summary.csv` for downstream use

### 5. Recommendation System
- Built a **segment-aware, cosine-similarity** recommender
- Features: average price, rating, battery life, camera, performance, design,
  and display ratings (all standardized)
- Given a selected product, returns the top-N most similar products, optionally
  restricted to the same segment

### 6. Application Development (Streamlit)
`app.py` has four tabs:
1. **📊 Cluster Insights** — segment metrics, price/segment charts, segment summary table
2. **🔍 Explore Products** — filterable product table + brand-level comparison
3. **🎯 Recommendations** — pick a product, get similarity-ranked recommendations
4. **📈 Insights & Reporting** — segmentation analysis, high/low performers,
   price-vs-performance trends, customer preference patterns, and data-driven
   takeaways — all computed live from the current CSVs

---

##  Running the App

Place `app.py`, `product_segments_summary.csv`, and `reviews_with_product_segments.csv`
in the same folder, then:

```bash
pip install streamlit plotly pandas scikit-learn
streamlit run app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`).

> **Note:** If you re-run the clustering/export cells in the notebook and update
> the CSVs, fully stop (`Ctrl+C`) and restart Streamlit rather than just
> refreshing the browser — cached data won't pick up file changes otherwise.

---

## Key Results

- 22 products segmented into 4 clusters (Budget, Mid-range, Upper-mid, Premium)
- Rating stays nearly flat (~3.1) across all price tiers — **price is not a
  strong predictor of customer satisfaction** in this dataset
- Spec-level ratings (camera, performance, battery) correlate much more
  strongly with overall rating than price does
- Working similarity-based recommender validated against multiple sample products
- Fully interactive Streamlit app for visualization, exploration, and
  recommendation

---

##  Tech Stack

Python · Pandas · NumPy · Scikit-learn (K-Means, StandardScaler,
Cosine Similarity) · Plotly · Streamlit