import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import date, datetime

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load("simple_house_price_model.pkl")
training_data = pd.read_csv("train.csv")
current_year = date.today().year

DEFAULTS = {
    "overall_qual": 7,
    "living_area": 1800,
    "garage_cars": 2,
    "garage_area": 500,
    "basement_area": 1000,
    "first_floor_area": 1100,
    "full_baths": 2,
    "year_built": 2005,
    "year_remodeled": 2005,
    "total_rooms": 7
}

PRESETS = {
    "Typical house": DEFAULTS,
    "Small house": {
        "overall_qual": 5, "living_area": 1200,
        "garage_cars": 1, "garage_area": 250,
        "basement_area": 500, "first_floor_area": 700,
        "full_baths": 1, "year_built": 1995,
        "year_remodeled": 1995, "total_rooms": 5
    },
    "Premium house": {
        "overall_qual": 9, "living_area": 3000,
        "garage_cars": 3, "garage_area": 800,
        "basement_area": 1800, "first_floor_area": 1700,
        "full_baths": 3, "year_built": 2015,
        "year_remodeled": 2020, "total_rooms": 10
    }
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.session_state.setdefault("prediction_history", [])

def apply_preset():
    values = PRESETS[st.session_state.preset]
    for key, value in values.items():
        st.session_state[key] = value

def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.preset = "Typical house"

def make_house(values):
    return pd.DataFrame([{
        "OverallQual": values["overall_qual"],
        "GrLivArea": values["living_area"],
        "GarageCars": values["garage_cars"],
        "GarageArea": values["garage_area"],
        "TotalBsmtSF": values["basement_area"],
        "1stFlrSF": values["first_floor_area"],
        "FullBath": values["full_baths"],
        "YearBuilt": values["year_built"],
        "YearRemodAdd": values["year_remodeled"],
        "TotRmsAbvGrd": values["total_rooms"]
    }])

st.title("🏠 House Price Predictor")
st.caption("Estimate a house price using a machine-learning model.")

st.sidebar.header("About this model")
st.sidebar.write(
    "This model was trained on Kaggle house-price data "
    "using the main property characteristics."
)
st.sidebar.warning(
    "This is an estimate, not a professional property appraisal."
)
st.sidebar.metric("Reference model R²", "0.8941")

st.sidebar.selectbox(
    "Example house",
    list(PRESETS.keys()),
    key="preset",
    on_change=apply_preset
)

st.sidebar.button(
    "Reset inputs",
    on_click=reset_inputs
)

comparison_mode = st.sidebar.checkbox(
    "Compare with another house"
)

st.header("House A details")

st.subheader("Quality and size")

col1, col2, col3 = st.columns(3)

with col1:
    st.slider(
        "Overall quality (1–10)",
        1, 10,
        key="overall_qual",
        help="Overall material and finish quality."
    )

with col2:
    st.number_input(
        "Living area (sq ft)",
        min_value=100,
        max_value=10000,
        step=50,
        key="living_area"
    )

with col3:
    st.slider(
        "Total rooms",
        1, 15,
        key="total_rooms"
    )

st.subheader("Garage and basement")

col1, col2, col3 = st.columns(3)

with col1:
    st.slider(
        "Garage capacity (cars)",
        0, 5,
        key="garage_cars"
    )

with col2:
    st.number_input(
        "Garage area (sq ft)",
        min_value=0,
        max_value=3000,
        step=25,
        key="garage_area"
    )

with col3:
    st.number_input(
        "Basement area (sq ft)",
        min_value=0,
        max_value=5000,
        step=50,
        key="basement_area"
    )

st.subheader("Bathrooms, age, and floor area")

col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "First-floor area (sq ft)",
        min_value=0,
        max_value=5000,
        step=50,
        key="first_floor_area"
    )

with col2:
    st.slider(
        "Full bathrooms",
        0, 5,
        key="full_baths"
    )

with col3:
    st.number_input(
        "Year built",
        1800,
        current_year,
        key="year_built"
    )

st.number_input(
    "Year remodeled",
    1800,
    current_year,
    key="year_remodeled"
)

if st.session_state.year_remodeled < st.session_state.year_built:
    st.error("Year remodeled cannot be earlier than year built.")
    st.stop()

if comparison_mode:
    st.divider()
    st.header("House B details")

    b_col1, b_col2, b_col3 = st.columns(3)

    with b_col1:
        b_quality = st.slider(
            "House B quality",
            1, 10, 7,
            key="b_quality"
        )

        b_living_area = st.number_input(
            "House B living area (sq ft)",
            100, 10000, 1800, 50,
            key="b_living_area"
        )

        b_total_rooms = st.slider(
            "House B total rooms",
            1, 15, 7,
            key="b_total_rooms"
        )

    with b_col2:
        b_garage_cars = st.slider(
            "House B garage capacity",
            0, 5, 2,
            key="b_garage_cars"
        )

        b_garage_area = st.number_input(
            "House B garage area (sq ft)",
            0, 3000, 500, 25,
            key="b_garage_area"
        )

        b_basement_area = st.number_input(
            "House B basement area (sq ft)",
            0, 5000, 1000, 50,
            key="b_basement_area"
        )

    with b_col3:
        b_first_floor_area = st.number_input(
            "House B first-floor area (sq ft)",
            0, 5000, 1100, 50,
            key="b_first_floor_area"
        )

        b_full_baths = st.slider(
            "House B full bathrooms",
            0, 5, 2,
            key="b_full_baths"
        )

        b_year_built = st.number_input(
            "House B year built",
            1800, current_year, 2005,
            key="b_year_built"
        )

        b_year_remodeled = st.number_input(
            "House B year remodeled",
            1800, current_year, 2005,
            key="b_year_remodeled"
        )

    if b_year_remodeled < b_year_built:
        st.error(
            "House B remodeled year cannot be earlier than its built year."
        )
        st.stop()

st.divider()

if st.button(
    "Predict price",
    type="primary",
    use_container_width=True
):
    house_a_values = {
        "overall_qual": st.session_state.overall_qual,
        "living_area": st.session_state.living_area,
        "garage_cars": st.session_state.garage_cars,
        "garage_area": st.session_state.garage_area,
        "basement_area": st.session_state.basement_area,
        "first_floor_area": st.session_state.first_floor_area,
        "full_baths": st.session_state.full_baths,
        "year_built": st.session_state.year_built,
        "year_remodeled": st.session_state.year_remodeled,
        "total_rooms": st.session_state.total_rooms
    }

    house_a = make_house(house_a_values)
    prediction = model.predict(house_a)[0]
    price_per_sqft = prediction / st.session_state.living_area

    st.session_state.prediction_history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Estimated price": round(prediction, 2),
        "Living area (sq ft)": st.session_state.living_area,
        "Quality": st.session_state.overall_qual,
        "Year built": st.session_state.year_built
    })

    st.success(f"Estimated price: ${prediction:,.2f}")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Estimated price",
            f"${prediction:,.0f}"
        )

    with result_col2:
        st.metric(
            "Estimated price per sq ft",
            f"${price_per_sqft:,.0f}"
        )

    if comparison_mode:
        house_b_values = {
            "overall_qual": b_quality,
            "living_area": b_living_area,
            "garage_cars": b_garage_cars,
            "garage_area": b_garage_area,
            "basement_area": b_basement_area,
            "first_floor_area": b_first_floor_area,
            "full_baths": b_full_baths,
            "year_built": b_year_built,
            "year_remodeled": b_year_remodeled,
            "total_rooms": b_total_rooms
        }

        house_b = make_house(house_b_values)
        prediction_b = model.predict(house_b)[0]

        st.subheader("House comparison")

        compare_col1, compare_col2 = st.columns(2)

        with compare_col1:
            st.metric(
                "House A estimated price",
                f"${prediction:,.0f}"
            )

        with compare_col2:
            st.metric(
                "House B estimated price",
                f"${prediction_b:,.0f}"
            )

        difference = prediction_b - prediction

        if difference >= 0:
            st.info(
                f"House B is estimated to be "
                f"${difference:,.0f} more expensive."
            )
        else:
            st.info(
                f"House B is estimated to be "
                f"${abs(difference):,.0f} less expensive."
            )

        comparison_chart = pd.DataFrame({
                   "House": ["House A", "House B"],
                     "Estimated price": [prediction, prediction_b]
        })

        st.subheader("Visual price comparison")

        st.bar_chart(
            comparison_chart.set_index("House")
        )

    st.subheader("Feature impact explanation")

    sensitivity_changes = {
        "OverallQual": ("Overall quality +1", 1),
        "GrLivArea": ("Living area +10%", 1.10),
        "GarageCars": ("Garage capacity +1", 1),
        "GarageArea": ("Garage area +10%", 1.10),
        "TotalBsmtSF": ("Basement area +10%", 1.10),
        "1stFlrSF": ("First-floor area +10%", 1.10),
        "FullBath": ("Full bathroom +1", 1),
        "YearBuilt": ("Year built +1", 1),
        "YearRemodAdd": ("Year remodeled +1", 1),
        "TotRmsAbvGrd": ("Total rooms +1", 1)
    }

    sensitivity_rows = []


    for feature, (label, change) in sensitivity_changes.items():
        changed_house = house_a.copy()

        if change == 1:
            changed_house[feature] = changed_house[feature] + 1
        else:
            changed_house[feature] = changed_house[feature] * change

        changed_prediction = model.predict(changed_house)[0]

        sensitivity_rows.append({
            "Feature": label,
            "Estimated price change": changed_prediction - prediction
        })

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity = sensitivity.sort_values(
        by="Estimated price change"
    )

    st.bar_chart(
        sensitivity.set_index("Feature")
    )

    st.caption(
        "This chart shows approximate what-if sensitivity, "
        "not guaranteed cause-and-effect."
    )


    st.subheader("Training price distribution")

    sale_prices = training_data["SalePrice"].dropna()

    counts, bin_edges = np.histogram(
        sale_prices,
        bins=20
    )

    bin_labels = [
        f"${bin_edges[i]:,.0f}–${bin_edges[i + 1]:,.0f}"
        for i in range(len(bin_edges) - 1)
    ]

    distribution = pd.DataFrame({
        "Price range": bin_labels,
        "Number of houses": counts
    })

    st.bar_chart(
        distribution.set_index("Price range")
    )

    st.caption(
        "This chart shows the distribution of prices "
        "in the training dataset."
    )


    st.subheader("Prediction history")

    history = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    report = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download prediction history",
        data=report,
        file_name="house_prediction_history.csv",
        mime="text/csv"
    )
    
if st.session_state.prediction_history:
    if st.button("Clear prediction history"):
        st.session_state.prediction_history = []
        st.rerun()

st.divider()
st.caption("Built with Python, scikit-learn, and Streamlit.")
