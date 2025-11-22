import numpy as np 

def generate_distance(n, mean, std, min_val, max_val):
    """
    Generate delivery distances (in miles) using a normal distribution,
    then clip them to realistic bounds.

    Parameters:
        n        (int)   - number of rows to generate
        mean     (float) - mean of the normal distribution
        std      (float) - standard deviation of the distribution
        min_val  (float) - minimum allowed distance
        max_val  (float) - maximum allowed distance

    Returns:
        np.ndarray of shape (n,) with distance values
    """
    distances = np.random.normal(mean, std, n) # generates n distances using normal distribution with given mean and stddev
    distances = np.clip(distances, min_val, max_val) # removing unrealistic distances by clipping to [min_val, max_val]
    return distances # returns a numpy array of shape

def generate_subtotal(n, mean, sigma):
    """
    Generate order subtotal values using a lognormal distribution.
    This matches real-world order prices, which are skewed:
    - most orders are around $15–$30
    - some orders go to $50–$100
    - very few are extremely small or extremely large

    Parameters:
        n      (int)   - number of rows to generate
        mean   (float) - mean of the underlying normal distribution (log-space)
        sigma  (float) - standard deviation of the log-space distribution

    Returns:
        np.ndarray of shape (n,) with subtotal values in dollars
    """
    subtotals = np.random.lognormal(mean, sigma, n) # generates n subtotals using lognormal distribution with given mean and sigma
    return subtotals 

def generate_weather(n, choices, probs):
    """
    Generate weather conditions using a categorical distribution.

    Parameters:
        n        (int)      - number of rows
        choices  (list)     - category names (e.g., ["clear","rain","snow"])
        probs    (list)     - probability for each category

    Returns:
        np.ndarray of shape (n,) with weather labels
    """
    return np.random.choice(choices, p = probs, size = n)

def generate_time_of_day(n, choices, probs):
    """
    Generate time-of-day labels using a categorical distribution.
    """
    return np.random.choice(choices, p = probs, size = n)

def generate_day_of_week(n, choices, probs):
    """
    Generate day-of-week labels (Mon–Sun).
    """
    return np.random.choice(choices, p = probs, size = n)

def generate_rating(n, min_rating, max_rating):
    """
    Generate ratings between min_val and max_val (inclusive).
    Ratings are uniform: each value 1–5 is equally likely.
    """
    return np.random.randint(min_rating, max_rating + 1, size = n)

def generate_wait_time(n, mean=10, std=4, min_val=0, max_val=30):
    """
    Generate restaurant wait times (in minutes) using a normal distribution,
    then clip them to realistic bounds.

    Parameters:
        n        (int)   - number of samples
        mean     (float) - average wait time
        std      (float) - variation in wait time
        min_val  (int)   - minimum wait time
        max_val  (int)   - maximum wait time

    Returns:
        np.ndarray of shape (n,) with wait time values
    """

    # 1. Draw values from a normal distribution
    wait = np.random.normal(mean, std, n)

    # 2. Clip values to a realistic range (no negative waits)
    wait = np.clip(wait, min_val, max_val)

    return wait

def generate_item_count(n, lam=3):
    """
    Generate item counts using a Poisson distribution.
    Most orders have 2–4 items, with occasional larger orders.

    Parameters:
        n    (int) - number of rows to generate
        lam  (int) - average items per order (lambda)

    Returns:
        np.ndarray of shape (n,) with item counts
    """
    return np.random.poisson(lam=lam, size=n)

def generate_messages_sent(n, lam=1.2):
    """
    Generate the number of messages sent by the customer.
    - 0 or 1 messages are most common
    - occasional 2+ messages

    Parameters:
        n    (int) - number of rows to generate
        lam  (float) - average messages per customer

    Returns:
        np.ndarray of shape (n,) with message counts
    """
    return np.random.poisson(lam=lam, size=n)

