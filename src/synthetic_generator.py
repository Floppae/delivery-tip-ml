import pandas as pd 
from src.config import *
from src.generate_features import * 
from src.correlations import compute_tip_percent

def generate_dataset(n=N_ROWS):
    """
    Build the full synthetic delivery tipping dataset.
    Returns a pandas DataFrame with all features generated.
    """

    data = {}

    data["distance_miles"] = generate_distance(
        n, DISTANCE_MEAN, DISTANCE_STD, DISTANCE_MIN, DISTANCE_MAX
    )

    data["order_subtotal"] = generate_subtotal(
        n, SUBTOTAL_MEAN, SUBTOTAL_SIGMA
    )

    data["wait_time_minutes"] = generate_wait_time(
        n, mean=10, std=4, min_val=0, max_val=30
    )


    data["weather"] = generate_weather(n, WEATHER_CHOICES, WEATHER_PROBS)
    data["time_of_day"] = generate_time_of_day(n, TIME_OF_DAY_CHOICES, TIME_OF_DAY_PROBS)
    data["day_of_week"] = generate_day_of_week(n, DAY_OF_WEEK_CHOICES, DAY_OF_WEEK_PROBS)


    data["communication_rating"] = generate_rating(n, RATING_MIN, RATING_MAX)

    data["item_count"] = generate_item_count(n, lam=3)
    data["messages_sent"] = generate_messages_sent(n, lam=1.2)

    # Convert to DataFrame 
    df = pd.DataFrame(data)
    df["tip_percent"] = compute_tip_percent(df) # compute tip percent based on correlations
    df["tip_amount"]  = df["order_subtotal"] * (df["tip_percent"] / 100) # compute tip amount in dollars

    return df