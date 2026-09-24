import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Global mobile review ",
    layout="wide",
)

# ==================================================================
# DATA LOADING (cached)
# ==================================================================
@st.cache_data
def load_data():
    product_df = pd.read_csv("product_segments_summary.csv")
    reviews_df = pd.read_csv("reviews_with_product_segments.csv")
    return product_df, reviews_df


@st.cache_data
def build_similarity(product_df, reviews_df):
    """Rebuild the full-spec feature matrix + cosine similarity matrix,
    same logic as the notebook's Section 5 (segment-aware recommender)."""
    spec_df = reviews_df.groupby(["brand", "model"]).agg(
        avg_price_usd=("price_usd", "mean"),
        avg_rating=("rating", "mean"),
        avg_battery_life=("battery_life_rating", "mean"),
        avg_camera=("camera_rating", "mean"),
        avg_performance=("performance_rating", "mean"),
        avg_design=("design_rating", "mean"),
        avg_display=("display_rating", "mean"),
        review_count=("review_id", "count"),
    ).reset_index()

    rec_df = spec_df.merge(
        product_df[["brand", "model", "segment"]], on=["brand", "model"], how="left"
    )

    feature_cols = [
        "avg_price_usd", "avg_rating", "avg_battery_life",
        "avg_camera", "avg_performance", "avg_design", "avg_display",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rec_df[feature_cols])

    sim_matrix = cosine_similarity(X_scaled)
    sim_df = pd.DataFrame(sim_matrix, index=rec_df["model"], columns=rec_df["model"])

    return rec_df, sim_df


def recommend_similar_products(rec_df, sim_df, selected_model, top_n=3, strict_segment=True):
    selected_model = selected_model.strip()
    if selected_model not in sim_df.index:
        return None, None

    target_info = rec_df[rec_df["model"] == selected_model].iloc[0]
    target_segment = target_info["segment"]

    if strict_segment:
        candidate_models = rec_df[rec_df["segment"] == target_segment]["model"]
    else:
        candidate_models = rec_df["model"]

    sim_scores = sim_df[selected_model].loc[candidate_models].sort_values(ascending=False)
    recommended = sim_scores.drop(index=selected_model, errors="ignore").head(top_n)

    rows = []
    for model_name, score in recommended.items():
        row = rec_df[rec_df["model"] == model_name].iloc[0]
        rows.append({
            "Recommended Model": row["model"],
            "Brand": row["brand"],
            "Similarity Score": round(score, 4),
            "Price (USD)": round(row["avg_price_usd"], 2),
            "Price Delta ($)": round(row["avg_price_usd"] - target_info["avg_price_usd"], 2),
            "Rating": round(row["avg_rating"], 2),
        })

    return target_info, pd.DataFrame(rows)


# ==================================================================
# LOAD DATA
# ==================================================================
product_df, reviews_df = load_data()
rec_df, sim_df = build_similarity(product_df, reviews_df)

SEGMENT_ORDER = ["Budget", "Mid-range", "Upper-mid", "Premium"]

# ==================================================================
# HEADER
# ==================================================================
st.title("Global Mobile Reviews Dashboard")
# st.caption(
#     "K-Means clustering on price & review volume, plus a segment-aware "
#     "cosine-similarity recommender built on product specifications."
# )

tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Insights", "🔍 Explore Products", "🎯 Recommendations","📈 Insights & Reporting"])

# ==================================================================
# TAB 1 — CLUSTER INSIGHTS
# ==================================================================
with tab1:
    st.subheader("Product Segments Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", len(product_df))
    col2.metric("Total Reviews", int(product_df["review_count"].sum()))
    col3.metric("Avg. Price (USD)", f"${product_df['avg_price_usd'].mean():.0f}")
    col4.metric("Avg. Rating", f"{product_df['avg_rating'].mean():.2f} ")

    st.markdown("---")

    # c1, c2 = st.columns(2)

    # with c1:
    fig1 = px.scatter(
            product_df, x="avg_price_usd", y="review_count", color="segment",
            hover_data=["brand", "model", "avg_rating"],
            category_orders={"segment": SEGMENT_ORDER},
            title="Product Clusters: Price vs. Review Count",
        )
    st.plotly_chart(fig1, use_container_width=True)

    # with c2:
    fig2 = px.box(
            product_df, x="segment", y="avg_price_usd", points="all", color="segment",
            category_orders={"segment": SEGMENT_ORDER},
            title="Price Distribution by Segment",
        )
    st.plotly_chart(fig2, use_container_width=True)

    # c3, c4 = st.columns(2)

    # with c3:
    seg_counts = product_df["segment"].value_counts().reindex(SEGMENT_ORDER).reset_index()
    seg_counts.columns = ["segment", "count"]
    fig3 = px.bar(
            seg_counts, x="segment", y="count", color="segment",
            category_orders={"segment": SEGMENT_ORDER},
            title="Number of Products per Segment",
        )
    st.plotly_chart(fig3, use_container_width=True)

    # with c4:
    fig4 = px.bar(
            product_df.sort_values("avg_rating", ascending=False),
            x="model", y="avg_rating", color="segment", orientation="v",
            category_orders={"segment": SEGMENT_ORDER},
            title="Products Ranked by Average Rating",
        )
    fig4.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Summary Table")

    segment_profile = product_df.groupby("segment").agg(
        total_models=("model", "count"),
        min_price=("avg_price_usd", "min"),
        avg_price=("avg_price_usd", "mean"),
        max_price=("avg_price_usd", "max"),
        total_reviews=("review_count", "sum"),
        avg_rating=("avg_rating", "mean"),
    ).round(2).reindex(SEGMENT_ORDER).reset_index()

    st.dataframe(segment_profile, use_container_width=True, hide_index=True)

# ==================================================================
# TAB 2 — EXPLORE PRODUCTS
# ==================================================================
with tab2:
    st.subheader("Browse Products by Segment")

    seg_filter = st.multiselect(
        "Filter by segment", options=SEGMENT_ORDER, default=SEGMENT_ORDER
    )
    brand_filter = st.multiselect(
        "Filter by brand", options=sorted(product_df["brand"].unique()),
        default=sorted(product_df["brand"].unique()),
    )

    filtered = product_df[
        product_df["segment"].isin(seg_filter) & product_df["brand"].isin(brand_filter)
    ].sort_values("avg_price_usd", ascending=False)

    st.dataframe(
        filtered[["brand", "model", "segment", "avg_price_usd", "avg_rating", "review_count"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.subheader("Brand-Level Comparison")

    brand_summary = reviews_df.groupby("brand").agg(
        avg_price=("price_usd", "mean"),
        avg_rating=("rating", "mean"),
        avg_camera=("camera_rating", "mean"),
        pct_positive=("sentiment", lambda s: (s == "Positive").mean() * 100),
        review_count=("review_id", "count"),
    ).round(2).sort_values("avg_rating", ascending=False).reset_index()

    st.dataframe(brand_summary, use_container_width=True, hide_index=True)

# ==================================================================
# TAB 3 — RECOMMENDATIONS
# ==================================================================
with tab3:
    st.subheader("Find Similar Products")

    all_models = sorted(rec_df["model"].unique())
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        selected_model = st.selectbox("Select a product", options=all_models)
    with c2:
        top_n = st.slider("Number of recommendations", 1, 5, 3)
    with c3:
        strict_segment = st.checkbox("Stay within same segment", value=True)

    if st.button("Get Recommendations", type="primary"):
        target_info, results = recommend_similar_products(
            rec_df, sim_df, selected_model, top_n=top_n, strict_segment=strict_segment
        )

        if results is None:
            st.error(f"'{selected_model}' not found.")
        else:
            st.markdown(
                f"**Target:** {target_info['brand']} {target_info['model']} "
                f"| Segment: **{target_info['segment']}** "
                f"| Price: **${target_info['avg_price_usd']:.2f}** "
                f"| Rating: **{target_info['avg_rating']:.2f}**"
            )
            st.dataframe(results, use_container_width=True, hide_index=True)

            fig = px.bar(
                results, x="Recommended Model", y="Similarity Score", color="Brand",
                title="Similarity Score of Recommended Products",
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# TAB 4 — INSIGHTS & REPORTING
# ==================================================================
with tab4:
    st.subheader("📈 Insights & Reporting")
    # st.caption("All figures below are computed live from the current CSV data.")
    st.markdown("  ")
    tab5,tab6,tab7,tab8 = st.tabs(["Product Segmentation", "High/Low Performing Products", "Price vs Performance", "Customer Preference Patterns"])
    # ---------- 1. Product Segmentation ----------
    with tab5:
        st.subheader("Product Segmentation (Cluster-wise Analysis)")
 
        segment_profile = product_df.groupby("segment").agg(
        total_models=("model", "count"),
        min_price=("avg_price_usd", "min"),
        avg_price=("avg_price_usd", "mean"),
        max_price=("avg_price_usd", "max"),
        total_reviews=("review_count", "sum"),
        avg_rating=("avg_rating", "mean"),
    ).round(2).reindex(SEGMENT_ORDER).reset_index()
        st.dataframe(segment_profile, use_container_width=True, hide_index=True)
 
        rating_spread = product_df["avg_rating"].max() - product_df["avg_rating"].min()
        st.info( 
           f"Ratings stay within a narrow **{rating_spread:.2f}-point band** across all four "
        # f"price segments — price tier alone does not strongly separate customer satisfaction."
    )
 
        fig_seg = px.bar(
        segment_profile, x="segment", y="avg_price", color="segment",
        category_orders={"segment": SEGMENT_ORDER},
        title="Average Price by Segment", text_auto=".0f",
    )
        st.plotly_chart(fig_seg, use_container_width=True)
 
        st.markdown("---")
 
    # ---------- 2. High / Low Performing Products ----------
    with tab6:
        st.subheader("High-Performing and Low-Performing Products") 
    # c1, c2 = st.columns(2)
    # with c1:
        st.write("**Top 5 by rating**")
        st.dataframe(
            product_df.sort_values("avg_rating", ascending=False).head(5)
            [["brand", "model", "segment", "avg_price_usd", "avg_rating"]],
            use_container_width=True, hide_index=True,
        )

    # with c2:
        top_5 = px.bar(
            product_df.sort_values("avg_rating", ascending=False).head(5),
            x="model", y="avg_rating", color="segment",
            category_orders={"segment": SEGMENT_ORDER},
            title="Top 5 Products by Rating",
        )
        st.plotly_chart(top_5, use_container_width=True)

    # c3,c4 = st.columns(2)
    # with c3:
        st.markdown("---")

        st.write("**Bottom 5 by rating**")
        st.dataframe(
            product_df.sort_values("avg_rating").head(5)
            [["brand", "model", "segment", "avg_price_usd", "avg_rating"]],
            use_container_width=True, hide_index=True,
        )

    # with c4:
        bottom_5 = px.bar(
            product_df.sort_values("avg_rating").head(5),
            x="model", y="avg_rating", color="segment",
            category_orders={"segment": SEGMENT_ORDER},
            title="Bottom 5 Products by Rating",
        )
        st.plotly_chart(bottom_5, use_container_width=True)


    # st.markdown("**Best value-for-money (rating per $100 spent)**")
    # value_df = product_df.copy()
    # value_df["value_score"] = (value_df["avg_rating"] / value_df["avg_price_usd"] * 100).round(4)
    # st.dataframe(
    #     value_df.sort_values("value_score", ascending=False).head(5)
    #     [["brand", "model", "segment", "avg_price_usd", "avg_rating", "value_score"]],
    #     use_container_width=True, hide_index=True,
    # )
 
    # fig_perf = px.bar(
    #     product_df.sort_values("avg_rating", ascending=False),
    #     x="model", y="avg_rating", color="segment",
    #     category_orders={"segment": SEGMENT_ORDER},
    #     title="All Products Ranked by Average Rating",
    # )
    # fig_perf.update_layout(xaxis_tickangle=-45)
    # st.plotly_chart(fig_perf, use_container_width=True)
 
        st.markdown("---")
 
    # ---------- 3. Price vs Performance ----------
    with tab7:
        st.subheader("Price vs. Performance Trends")
 
        spec_cols = ["rating", "camera_rating", "performance_rating",
                 "battery_life_rating", "display_rating", "design_rating"]
        corr_with_price = reviews_df[["price_usd"] + spec_cols].corr()["price_usd"].drop("price_usd").round(3)
        corr_with_rating = reviews_df[spec_cols].corr()["rating"].drop("rating").round(3)
 
    # c3, c4 = st.columns(2)
    # with c3:
        st.write("**Correlation: Price vs. spec ratings**")
        st.dataframe(corr_with_price.rename("correlation").reset_index().rename(columns={"index": "metric"}),
                     use_container_width=True, hide_index=True)
    # with c4:
        st.write("**Correlation: Overall rating vs. spec ratings**")
        st.dataframe(corr_with_rating.rename("correlation").reset_index().rename(columns={"index": "metric"}),
                     use_container_width=True, hide_index=True)
 
        price_rating_corr = reviews_df[["price_usd", "rating"]].corr().iloc[0, 1]
        st.info(
        f"Price correlates with overall rating at just **r = {price_rating_corr:.3f}**  "
        # f"almost no relationship — while spec ratings (camera, performance, battery, etc.) "
        # f"correlate much more strongly with overall rating. **Customers respond to spec quality, "
        # f"not price.**"
        )
 
        reviews_df["price_quintile"] = pd.qcut(reviews_df["price_usd"], 5, duplicates="drop")
        quintile_summary = reviews_df.groupby("price_quintile", observed=True).agg(
        avg_rating=("rating", "mean"),
        avg_camera=("camera_rating", "mean"),
        avg_performance=("performance_rating", "mean"),
        n_reviews=("review_id", "count"),
        ).round(2).reset_index()
        quintile_summary["price_quintile"] = quintile_summary["price_quintile"].astype(str)
 
        fig_price_perf = px.line(
        quintile_summary, x="price_quintile", y="avg_rating", markers=True,
        title="Average Rating Across Price Quintiles",
        )
        st.plotly_chart(fig_price_perf, use_container_width=True)
 
    # st.markdown("---")
 
    # ---------- 4. Customer Preference Patterns ----------
    with tab8:
        st.subheader("Customer Preference Patterns")
 
    # c5, c6 = st.columns(2)
    # with c5:

        st.write("**Sentiment mix by segment (%)**")
        sentiment_mix = (
            pd.crosstab(reviews_df["segment"], reviews_df["sentiment"], normalize="index") * 100
        ).round(1).reindex(SEGMENT_ORDER).reset_index()
        st.dataframe(sentiment_mix, use_container_width=True, hide_index=True)
 
    # with c6:
        st.write("**Average rating by age group**")
        age_summary = reviews_df.groupby("age_group", observed=True).agg(
            avg_rating=("rating", "mean"),
            avg_helpful_votes=("helpful_votes", "mean"),
            n_reviews=("review_id", "count"),
        ).round(2).reset_index()
        st.dataframe(age_summary, use_container_width=True, hide_index=True)
 
        fig_country = px.bar(
        reviews_df["country"].value_counts().head(10).reset_index(),
        x="country", y="count", title="Top 10 Countries by Review Volume",
        )
        st.plotly_chart(fig_country, use_container_width=True)
 
        top_country_by_segment = (
        reviews_df.groupby(["segment", "country"]).size().reset_index(name="n")
        .sort_values("n", ascending=False).groupby("segment").first().reindex(SEGMENT_ORDER).reset_index()
        )
        st.write("**Leading country per segment**")
        st.dataframe(top_country_by_segment, use_container_width=True, hide_index=True)
 
        # st.markdown("---")
