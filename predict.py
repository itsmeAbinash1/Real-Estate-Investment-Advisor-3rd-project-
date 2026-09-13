import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import mlflow

# ============================================================
# CONFIG - edit these two paths if your files live elsewhere
# ============================================================
DATA_PATH = r"C:/Users/HP/Downloads/india_housing_prices.csv"        # your raw CSV
ARTIFACT_DIR = "streamlit_artifacts"            # from the notebook export cells
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"     # must match your notebook exactly

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# DATA LOADING (for the Introduction / EDA pages)
# Applies the same cleaning steps your notebook applied, so the
# units/values here match what the models were actually trained on.
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert lakhs to rupees (matches notebook cell 8)
    df["Price_in_Lakhs"] = (df["Price_in_Lakhs"].astype("float64") * 100000).astype("int64")

    # Fix rows where Total_Floors < Floor_No (matches notebook cell 10)
    invalid_rows = df["Total_Floors"] < df["Floor_No"]
    df.loc[invalid_rows, "Total_Floors"] = df.loc[invalid_rows, "Floor_No"]

    # Recalculate Price_per_SqFt from the corrected price (matches notebook cell 20)
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

    return df


# ============================================================
# CACHED MODEL LOADERS (for the Prediction page)
# ============================================================
@st.cache_resource
def load_classification_model():
    return mlflow.sklearn.load_model("models:/GoodInvestment_Classifier@champion")


@st.cache_resource
def load_regression_model():
    return mlflow.sklearn.load_model("models:/FuturePrice5Y_Regressor@champion")


@st.cache_resource
def load_prediction_artifacts():
    scaler_clf = joblib.load(f"{ARTIFACT_DIR}/scaler_clf.joblib")
    scaler_reg = joblib.load(f"{ARTIFACT_DIR}/scaler_reg.joblib")

    with open(f"{ARTIFACT_DIR}/classification_features.json") as f:
        clf_features = json.load(f)
    with open(f"{ARTIFACT_DIR}/regression_features.json") as f:
        reg_features = json.load(f)
    with open(f"{ARTIFACT_DIR}/dropdown_options.json") as f:
        dropdowns = json.load(f)
    with open(f"{ARTIFACT_DIR}/locality_frequency.json") as f:
        locality_freq = json.load(f)
    with open(f"{ARTIFACT_DIR}/defaults.json") as f:
        defaults = json.load(f)
    with open(f"{ARTIFACT_DIR}/model_meta.json") as f:
        model_meta = json.load(f)
    with open(f"{ARTIFACT_DIR}/city_to_state.json") as f:
        city_to_state = json.load(f)
    with open(f"{ARTIFACT_DIR}/hidden_defaults.json") as f:
        hidden_defaults = json.load(f)

    return (scaler_clf, scaler_reg, clf_features, reg_features, dropdowns,
            locality_freq, defaults, model_meta, city_to_state, hidden_defaults)


REFERENCE_YEAR = 2025  # matches Age_of_Property's reference year in the notebook


def build_raw_row(visible, hidden, city_to_state):
    """
    Merge the 9 user-visible inputs with the hidden defaults into one
    raw (pre-one-hot) row, exactly matching the notebook's feature
    engineering. Amenities defaults to none selected (Amenity_Count=0)
    since it isn't shown on this simplified form.
    """
    row = {
        "BHK": visible["BHK"],
        "Size_in_SqFt": visible["Size_in_SqFt"],
        "Price_in_Lakhs": visible["Price_in_Lakhs"],
        "Price_per_SqFt": (visible["Price_in_Lakhs"] / visible["Size_in_SqFt"]) if visible["Size_in_SqFt"] else 0,
        "Nearby_Schools": visible["Nearby_Schools"],
        "Nearby_Hospitals": visible["Nearby_Hospitals"],
        "Parking_Space": 1 if visible["Parking_Space"] else 0,

        # Hidden fields - already-encoded 0/1 values, injected directly
        "Security": hidden["Security_encoded"],
        "Availability_Status": hidden["Availability_Status_encoded"],
        "Year_Built": hidden["Year_Built"],
        "Floor_No": hidden["Floor_No"],
        "Total_Floors": hidden["Total_Floors"],
        "Age_of_Property": REFERENCE_YEAR - hidden["Year_Built"],

        # No amenities exposed on this form -> neutral default (none selected)
        "Playground": 0, "Gym": 0, "Garden": 0, "Pool": 0, "Clubhouse": 0,
        "Amenity_Count": 0,

        # Categorical columns - a mix of visible and hidden, all kept as
        # raw text so pd.get_dummies() reproduces the exact training columns
        "City": visible["City"],
        "State": city_to_state.get(visible["City"], "Unknown"),
        "Property_Type": visible["Property_Type"],
        "Public_Transport_Accessibility": visible["Public_Transport_Accessibility"],
        "Furnished_Status": hidden["Furnished_Status"],
        "Facing": hidden["Facing"],
        "Owner_Type": hidden["Owner_Type"],
    }
    return pd.DataFrame([row])


def build_model_input(visible, hidden, feature_list, locality_freq, defaults, city_to_state, is_regression):
    raw_df = build_raw_row(visible, hidden, city_to_state)

    # Locality isn't on this form either -> use the dataset-wide median
    # frequency, same fallback used elsewhere.
    raw_df["Locality_Frequency"] = defaults["median_locality_freq"]

    categorical_cols = ["Furnished_Status", "Property_Type",
                         "Public_Transport_Accessibility", "Facing", "Owner_Type"]

    if is_regression:
        # Regressor was trained with one-hot City_*, not raw State.
        categorical_cols.append("City")
        raw_df = raw_df.drop(columns=["State"])
    else:
        # Classifier was trained without City/State at all.
        raw_df = raw_df.drop(columns=["City", "State"])

    encoded = pd.get_dummies(raw_df, columns=categorical_cols)

    # Reindex to the exact training column list/order; anything the model
    # expects that wasn't produced (e.g. an unselected City_*) becomes 0,
    # which is correct for one-hot columns.
    return encoded.reindex(columns=feature_list, fill_value=0)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.title("Navigation")
    st.write("Go to:")
    page = st.radio("", ["Introduction", "EDA Visualizations", "Prediction"])


# ============================================================
# PAGE 1: INTRODUCTION
# ============================================================
if page == "Introduction":

    st.title("Real Estate Investment Advisor")
    st.subheader("Predicting Property Profitability & Future Value")

    st.header("Project Overview")
    st.write("This application assists real estate investors by:")
    st.markdown("""
    - **Classifying** whether a property is a *Good Investment*
    - **Predicting** the *Estimated Property Price After 5 Years*
    - Providing **interactive EDA visualizations**
    - Supporting **intelligent decision-making** for buyers, sellers, and investors
    """)

    st.header("Skills Used")
    st.write("Python • Pandas • NumPy • Matplotlib • Seaborn • Scikit-learn • Machine Learning • Streamlit")


# ============================================================
# PAGE 2: EDA VISUALIZATIONS
# ============================================================
elif page == "EDA Visualizations":

    df = load_data()

    st.title("EDA Visualizations")
    st.write("Explore the dataset through interactive visualizations.")

    # Q1
    st.subheader("Q1. What is the distribution of property prices?")
    fig = px.histogram(df, x="Price_in_Lakhs", nbins=50,
                        title="Distribution of Property Prices",
                        labels={"Price_in_Lakhs": "Property Price (₹)"})
    st.plotly_chart(fig, use_container_width=True)
    st.write("This histogram shows how property prices are distributed across the dataset.")

    # Q2
    st.subheader("Q2. What is the distribution of property sizes?")
    fig = px.histogram(df, x="Size_in_SqFt", nbins=50,
                        title="Distribution of Property Sizes",
                        labels={"Size_in_SqFt": "Size (SqFt)"})
    st.plotly_chart(fig, use_container_width=True)
    st.write("This graph shows the distribution of residential property sizes in square feet.")

    # Q3
    st.subheader("Q3. How does Price per SqFt vary by property type?")
    fig = px.box(df, x="Property_Type", y="Price_per_SqFt",
                 title="Price per SqFt by Property Type",
                 labels={"Property_Type": "Property Type", "Price_per_SqFt": "Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)
    st.write("The box plot compares the distribution of price per square foot among different property types.")

    # Q4
    st.subheader("Q4. Is there a relationship between property size and price?")
    fig = px.scatter(df.sample(min(10000, len(df))), x="Size_in_SqFt", y="Price_in_Lakhs",
                      color="Property_Type", title="Property Size vs Property Price",
                      labels={"Size_in_SqFt": "Size (SqFt)", "Price_in_Lakhs": "Property Price (₹)"})
    st.plotly_chart(fig, use_container_width=True)
    st.write("This scatter plot helps identify the relationship between property size and price.")

    # Q5
    st.subheader("Q5. Are there outliers in price, size and price per SqFt?")
    outlier_df = df[["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]]
    fig = go.Figure()
    fig.add_trace(go.Box(y=outlier_df["Price_in_Lakhs"], name="Property Price"))
    fig.add_trace(go.Box(y=outlier_df["Size_in_SqFt"], name="Property Size"))
    fig.add_trace(go.Box(y=outlier_df["Price_per_SqFt"], name="Price per SqFt"))
    fig.update_layout(title="Outlier Analysis", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Box plots are used to identify unusually high or low observations.")

    # Q6
    st.subheader("Q6. Which states have the highest average Price per SqFt?")
    state_price = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(state_price, x="State", y="Price_per_SqFt",
                 title="Average Price per SqFt by State",
                 labels={"State": "State", "Price_per_SqFt": "Average Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q7
    st.subheader("Q7. Which cities have the highest average property prices?")
    city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(city_price, x="City", y="Price_in_Lakhs",
                 title="Average Property Price by City",
                 labels={"City": "City", "Price_in_Lakhs": "Average Property Price (₹)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Q8
    st.subheader("Q8. What is the median property age across localities?")
    locality_age = df.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False).head(20).reset_index()
    fig = px.bar(locality_age, x="Locality", y="Age_of_Property",
                 title="Top 20 Localities by Median Property Age",
                 labels={"Locality": "Locality", "Age_of_Property": "Median Property Age (Years)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Q9
    st.subheader("Q9. How are BHK properties distributed across cities?")
    bhk_city = df.groupby(["City", "BHK"]).size().reset_index(name="Property_Count")
    fig = px.bar(bhk_city, x="City", y="Property_Count", color="BHK", barmode="stack",
                 title="BHK Distribution Across Cities",
                 labels={"City": "City", "Property_Count": "Number of Properties", "BHK": "BHK"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Q10
    st.subheader("Q10. Which are the top 5 most expensive localities?")
    top_localities = df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).reset_index()
    fig = px.bar(top_localities, x="Locality", y="Price_in_Lakhs",
                 title="Top 5 Most Expensive Localities",
                 labels={"Locality": "Locality", "Price_in_Lakhs": "Average Property Price (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q11
    st.subheader("Q11. What is the correlation between numerical features?")
    numerical_columns = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
                          "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]
    correlation_matrix = df[numerical_columns].corr()
    fig = px.imshow(correlation_matrix, text_auto=".2f", aspect="auto",
                     title="Correlation Heatmap of Numerical Features",
                     labels={"x": "Features", "y": "Features", "color": "Correlation"})
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "The correlation heatmap shows the strength and direction of "
        "relationships between numerical features. Values close to +1 "
        "indicate a strong positive relationship, values close to -1 "
        "indicate a strong negative relationship, and values close to 0 "
        "indicate a weak or no linear relationship."
    )

    # Q12
    st.subheader("Q12. Does the number of nearby schools affect Price per SqFt?")
    fig = px.scatter(df.sample(min(10000, len(df))), x="Nearby_Schools", y="Price_per_SqFt",
                      title="Nearby Schools vs Price per SqFt",
                      labels={"Nearby_Schools": "Nearby Schools", "Price_per_SqFt": "Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q13
    st.subheader("Q13. Does the number of nearby hospitals affect Price per SqFt?")
    fig = px.scatter(df.sample(min(10000, len(df))), x="Nearby_Hospitals", y="Price_per_SqFt",
                      title="Nearby Hospitals vs Price per SqFt",
                      labels={"Nearby_Hospitals": "Nearby Hospitals", "Price_per_SqFt": "Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q14
    st.subheader("Q14. How does furnished status affect property price?")
    fig = px.box(df, x="Furnished_Status", y="Price_in_Lakhs",
                 title="Property Price by Furnished Status",
                 labels={"Furnished_Status": "Furnished Status", "Price_in_Lakhs": "Property Price (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q15
    st.subheader("Q15. How does facing direction affect Price per SqFt?")
    fig = px.box(df, x="Facing", y="Price_per_SqFt",
                 title="Price per SqFt by Facing Direction",
                 labels={"Facing": "Facing Direction", "Price_per_SqFt": "Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q16
    st.subheader("Q16. How many properties belong to each owner type?")
    owner_count = df["Owner_Type"].value_counts().reset_index()
    owner_count.columns = ["Owner_Type", "Property_Count"]
    fig = px.bar(owner_count, x="Owner_Type", y="Property_Count",
                 title="Property Count by Owner Type",
                 labels={"Owner_Type": "Owner Type", "Property_Count": "Number of Properties"})
    st.plotly_chart(fig, use_container_width=True)

    # Q17
    st.subheader("Q17. How many properties are available in each status?")
    availability_count = df["Availability_Status"].value_counts().reset_index()
    availability_count.columns = ["Availability_Status", "Property_Count"]
    fig = px.pie(availability_count, names="Availability_Status", values="Property_Count",
                 title="Property Distribution by Availability Status")
    st.plotly_chart(fig, use_container_width=True)

    # Q18
    st.subheader("Q18. Does parking availability affect property price?")
    fig = px.box(df, x="Parking_Space", y="Price_in_Lakhs",
                 title="Property Price by Parking Availability",
                 labels={"Parking_Space": "Parking Available", "Price_in_Lakhs": "Property Price (₹)"})
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "This box plot compares property prices based on parking availability. "
        "It helps identify whether properties with parking tend to have "
        "higher or lower prices compared with properties without parking."
    )

    # Q19
    st.subheader("Q19. Does the number of amenities affect Price per SqFt?")
    amenities = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
    df["Amenity_Count"] = df["Amenities"].apply(
        lambda x: sum(amenity.lower() in str(x).lower() for amenity in amenities)
    )
    amenity_price = df.groupby("Amenity_Count")["Price_per_SqFt"].mean().reset_index()
    fig = px.bar(amenity_price, x="Amenity_Count", y="Price_per_SqFt",
                 title="Average Price per SqFt by Number of Amenities",
                 labels={"Amenity_Count": "Number of Amenities", "Price_per_SqFt": "Average Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # Q20
    st.subheader("Q20. How does public transport accessibility affect Price per SqFt?")
    transport_price = (
        df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"]
        .mean().reset_index().sort_values("Price_per_SqFt", ascending=False)
    )
    fig = px.bar(transport_price, x="Public_Transport_Accessibility", y="Price_per_SqFt",
                 title="Public Transport Accessibility vs Price per SqFt",
                 labels={"Public_Transport_Accessibility": "Public Transport Accessibility",
                         "Price_per_SqFt": "Average Price per SqFt (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    st.success("EDA Questions 1–20 completed successfully.")


# ============================================================
# PAGE 3: PREDICTION
# ============================================================
elif page == "Prediction":

    st.title("Property Investment Prediction")
    st.write("Fill out the form below to view investment classification and price forecast.")

    try:
        classifier = load_classification_model()
        regressor = load_regression_model()
        (scaler_clf, scaler_reg, clf_features, reg_features, dropdowns,
         locality_freq, defaults, model_meta, city_to_state, hidden_defaults) = load_prediction_artifacts()
        ready = True
    except Exception as e:
        ready = False
        st.error(
            "Could not load models or artifacts. Check that:\n\n"
            "1. MLFLOW_TRACKING_URI matches your notebook's `sqlite:///mlflow.db`, "
            "and Streamlit is running from the same folder that file is in\n"
            "2. You ran BOTH export cells in the notebook (the main one and "
            "the hidden-defaults one) - together they create everything in "
            "`streamlit_artifacts/` and set the `champion` alias on both models\n\n"
            f"Error detail: `{e}`"
        )

    if ready:
        row1 = st.columns(3)
        with row1[0]:
            city = st.selectbox("City", dropdowns["City"])
        with row1[1]:
            size_sqft = st.number_input("Size (Sq Ft)", min_value=100, value=1000, step=50)
        with row1[2]:
            nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, value=0, step=1)

        row2 = st.columns(3)
        with row2[0]:
            property_type = st.selectbox("Property Type", dropdowns["Property_Type"])
        with row2[1]:
            price_lakhs = st.number_input("Current Price (Lakhs)", min_value=1.0, value=50.0, step=1.0)
        with row2[2]:
            transport_access = st.selectbox(
                "Public Transport Accessibility", dropdowns["Public_Transport_Accessibility"]
            )

        row3 = st.columns(3)
        with row3[0]:
            bhk = st.number_input("BHK", min_value=1, max_value=10, value=1, step=1)
        with row3[1]:
            nearby_schools = st.number_input("Nearby Schools", min_value=0, value=0, step=1)
        with row3[2]:
            parking = st.selectbox("Parking Space", ["No", "Yes"]) == "Yes"

        if st.button("Predict", type="primary"):
            visible_inputs = {
                "City": city, "Size_in_SqFt": size_sqft, "Nearby_Hospitals": nearby_hospitals,
                "Property_Type": property_type, "Price_in_Lakhs": price_lakhs,
                "Public_Transport_Accessibility": transport_access,
                "BHK": bhk, "Nearby_Schools": nearby_schools, "Parking_Space": parking,
            }

            try:
                clf_input = build_model_input(visible_inputs, hidden_defaults, clf_features,
                                               locality_freq, defaults, city_to_state, is_regression=False)
                reg_input = build_model_input(visible_inputs, hidden_defaults, reg_features,
                                               locality_freq, defaults, city_to_state, is_regression=True)

                clf_input_final = scaler_clf.transform(clf_input) if model_meta["classifier_needs_scaling"] else clf_input
                reg_input_final = scaler_reg.transform(reg_input) if model_meta["regressor_needs_scaling"] else reg_input

                pred_class = classifier.predict(clf_input_final)[0]
                pred_proba = (
                    classifier.predict_proba(clf_input_final)[0][1]
                    if hasattr(classifier, "predict_proba") else None
                )
                pred_price = float(regressor.predict(reg_input_final)[0])

                if pred_class == 1:
                    st.success("Investment Decision: **Good Investment**")
                else:
                    st.error("Investment Decision: **Not a Good Investment**")

                if pred_proba is not None:
                    st.info(f"Model Confidence: {pred_proba * 100:.1f}%")

                st.subheader("Estimated Future Price (5 Years)")
                st.write(f"₹ {pred_price:,.2f} Lakhs")

                current_price_rupees = price_lakhs * 100000
                growth_pct = ((pred_price - current_price_rupees) / current_price_rupees) * 100
                st.metric("Price Growth (%)", f"{growth_pct:.1f}%")

            except Exception as e:
                st.error(f"Prediction failed: `{e}`")
                st.info(
                    "If this mentions missing columns or mismatched feature names, "
                    "re-run both export cells in the notebook - the saved feature "
                    "lists no longer match the currently registered champion models."
                )