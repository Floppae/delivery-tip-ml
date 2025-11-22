# -----------------------------------------
# Global configuration for synthetic dataset
# Defining parameters for the synthetic dataset distributions
# -----------------------------------------------------------

# Number of rows to generate
N_ROWS = 2000

# -------------------------------------------------
# Distance (miles)
# - Most deliveries are between 1–6 miles
# - Normal distribution with clipping to [0.1, 15]
# -------------------------------------------------
DISTANCE_MEAN = 3.5
DISTANCE_STD = 2.0 # some orders are very close while some can be very far, so a larger stddev creates more variance
DISTANCE_MIN = 0.1
DISTANCE_MAX = 15.0

# -------------------------------------------------
# Order Subtotal ($)
# - Typically skewed right
# - Lognormal fits real-world orders well since most orders are small but some can be very large
# -------------------------------------------------
SUBTOTAL_MEAN = 3       # log-space mean
SUBTOTAL_SIGMA = 0.5    # spread of lognormal distribution

# -------------------------------------------------
# Weather (categorical)
# -------------------------------------------------
WEATHER_CHOICES = ["clear", "rain", "snow"]
WEATHER_PROBS = [0.7, 0.2, 0.1]

# -------------------------------------------------
# Time of day (categorical)
# -------------------------------------------------
TIME_OF_DAY_CHOICES = ["morning", "afternoon", "night"]
TIME_OF_DAY_PROBS = [0.2, 0.5, 0.3]

# -------------------------------------------------
# Day of week (categorical)
# -------------------------------------------------
DAY_OF_WEEK_CHOICES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
DAY_OF_WEEK_PROBS = [1/7]*7   # equal probability

# -------------------------------------------------
# Ratings (1–5 stars)
# -------------------------------------------------
RATING_MIN = 1
RATING_MAX = 5
