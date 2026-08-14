import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple



def extract_timestamp(
        tweet_ids: pd.Series
        ) -> pd.Series:
    """
    Derives datetime timestamps from Twitter message IDs.

    Parameters:
        tweet_ids (pd.Series): A pandas Series containing Twitter Snowflake IDs (numeric or string format).

    Returns:
        pd.Series: A pandas Series of datetime objects representing the creation time of each tweet.
    """
    # Twitter Snowflake epoch offset (Nov 04, 2010 UTC in ms)
    TWITTER_EPOCH = 1288834974657

    # Convert IDs to numeric integers (invalid values -> NaN)
    numeric_ids = pd.to_numeric(tweet_ids, errors='coerce')

    # Integer division by 2^22 (4194304) replaces bit-shift >> 22
    timestamps_ms = (numeric_ids // 4194304) + TWITTER_EPOCH

    # Convert millisecond timestamps to datetime objects
    return pd.to_datetime(timestamps_ms, unit='ms')



def get_majority_vote(label_array):

    """
    Aggregates multiple annotator labels into a single consensus label via majority vote.
    
    Aggregates multiple annotator labels into a single consensus label via majority vote.

    Parameters:
        label_array : array-like
            Array, list, or iterable containing the labels provided by multiple annotators for a single data instance.

    Returns:
        int or float
            Consensus label as an integer if a clear winner exists. Returns `np.nan` if there is a tie between top candidates.
    """

    # Count occurrences of each label
    counts = Counter(label_array)
    
    # Find highest number of votes of each unique label
    max_count = max(counts.values())
    
    # Identify label/labels with the highest vote 
    top_candidates = [label for label, count in counts.items() if count == max_count]
    
    # If there is more than 1 candidate, return NaN
    if len(top_candidates) > 1:
        return np.nan

    # Get the clear winner
    winner = int(top_candidates[0])
    
    # Check that the winning label falls within the expected category range (0 to 5)
    #if not (0 <= winner <= 5):
        #return np.nan
        
    # Otherwise return the clear winner
    return int(top_candidates[0])



def plot_tweet_length_distribution(
    df: pd.DataFrame,
    x_col: str = 'num_symbols_origmsg',
    hue_col: str = 'target',
    target_mapping: Optional[Dict[int, str]] = None,
    hue_order: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plots relative KDE distribution of tweet lengths per target class.

    Parameters:
        df : pd.DataFrame
            DataFrame containing the dataset.
        x_col : str, default='num_symbols_origmsg'
            Column name contaning the number of symbols in the tweet.
        hue_col : str, default='target'
            Column name used for grouping/hue categorization.
        target_mapping : dict, optional
            Dictionary with mapping integer target values to respective string names.

    Returns:
        plt.Figure
            Matplotlib Figure object.
    """

    # Create a local copy to safely map strings without mutating the original dataframe
    plot_df = df.copy()
    
    if target_mapping is not None and hue_col in plot_df.columns:
        plot_df[hue_col] = plot_df[hue_col].map(target_mapping)

    if hue_order is None:
        hue_order = list(target_mapping.values())

    #plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.kdeplot(
        data=plot_df,
        x=x_col,
        hue=hue_col,
        hue_order=hue_order,
        common_norm=False,  # minority classes scaled up to better compare shapes
        palette='tab10',
        fill=True,
        alpha=0.3, 
        ax=ax
    )

    mean_val = plot_df[x_col].mean()
    median_val = plot_df[x_col].median()

    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Average: {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle='solid', linewidth=2, label=f'Median: {median_val:.2f}')

    _, y_max = ax.get_ylim()

    ax.text(
        mean_val + 2,
        y_max * 0.9,
        f'mean={mean_val:.2f}',
        color='red',
        fontweight='bold',
    )
    ax.text(
        median_val - 2,
        y_max * 0.9,
        f'median={median_val:.2f}',
        color='darkorange',
        fontweight='bold',
        ha='right',
    )

    ax.set_title('Relative Tweet Length Distribution per Target Class (Normalized Density)')
    ax.set_xlabel('Num. of symbols in original tweet')
    ax.set_ylabel('Density')

    fig.tight_layout()
    
    return fig



def plot_tweet_volume_over_time(
    df: pd.DataFrame,
    date_col: str = 'created_at',
    resample_freq: str = 'D',
    figsize: Tuple[int, int] = (14, 6),
    color: str = 'royalblue'
) -> plt.Figure:
    """
    Plots a line chart representing tweet volume over time aggregated by a specified frequency.
    Aggregation options: 'D'-daily, 'W'-weekly, 'ME'-monthly (Month End)

    Parameters:
        df : pd.DataFrame
            Cleaned DataFrame containing the timestamp column.
        date_col : str, default='created_at'
            The name of the column containing timestamp/date data.
        resample_freq : str, default='D'
            The frequency for resampling time intervals (e.g., 'D' for daily, 'W' for weekly, 'ME' for monthly).
        figsize : tuple, default=(14, 6)
            The width and height of the figure.
        color : str, default='royalblue'
            The color of the line plot.

    Returns:
        plt.Figure
            Matplotlib Figure object
    """
    # Create a copy to avoid mutating the original DataFrame
    df_copy = df.copy()
    
    # Convert timestamp to standard pandas datetime objects
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Set date column as index and resample to count tweet volume
    tweet_counts = (
        df_copy.set_index(date_col)
        .resample(resample_freq)
        .size()
    )

    # Create aggregation map
    aggr_map = {'D': 'Date', 'W': 'Aggregated by Week', 'ME': 'Aggregated by Month'}

    # Create the figure and axes using the object-oriented approach
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=tweet_counts, 
        color=color, 
        linewidth=2,
        ax=ax
    )

    ax.set_title('Tweet Volume Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel(aggr_map[resample_freq], fontsize=12)
    ax.set_ylabel('Number of Tweets', fontsize=12)

    # Rotate x-axis date labels for better readability
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()

    return fig



def plot_hate_speech_proportions(
    df: pd.DataFrame,
    time_dimension: str = 'hour',
    date_col: str = 'created_at',
    target_col: str = 'target',
    target_mapping: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plots a chart showing the proportion of hate speech categories 
    across hours of the day or days of the week.

    Parameters:
        df : pd.DataFrame
            Cleaned DataFrame containing the dataset.
        time_dimension : str, default='hour'
            Time dimension to analyze. 
            Options are 'hour' (stacked area chart) or 'day' (stacked bar chart).
        date_col : str, default='created_at'
            Name of the column with timestamp data.
        target_col : str, default='target'
            Name of the column containing the category target labels.
        target_mapping : dict, optional
            A dictionary mapping integer target values to descriptive string names 
            for the legend. The order of keys/values dictates the legend stack order.
        figsize : tuple, default=(12, 6)
            The width and height of the figure.

    Returns:
        plt.Figure
            Matplotlib Figure object.
    """
    plot_df = df.copy()
    
    # Convert timestamp to standard pandas datetime type
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Apply target mapping and derive class order, if provided
    class_order = None
    if target_mapping is not None and target_col in plot_df.columns:
        plot_df[target_col] = plot_df[target_col].map(target_mapping)
        class_order = list(target_mapping.values())

    # Prepare data for plot: dimensions (hour, day), grouping
    dimension_lower = time_dimension.lower()
    
    if dimension_lower == 'hour':
        plot_df['time_bucket'] = plot_df[date_col].dt.hour
        counts = plot_df.groupby(['time_bucket', target_col]).size().unstack(fill_value=0)
        proportions = counts.div(counts.sum(axis=1), axis=0)
        
        chart_kind = 'area'
        title = 'Proportion of Hate Speech Categories Across Hours of the Day'
        xlabel = 'Hour of the Day (0 - 23)'
        xticks = range(24)
        rotation = 0
        
    elif dimension_lower in ['day', 'day_of_week']:
        plot_df['time_bucket'] = plot_df[date_col].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        counts = plot_df.groupby(['time_bucket', target_col]).size().unstack(fill_value=0)
        counts = counts.reindex(days_order, fill_value=0)
        proportions = counts.div(counts.sum(axis=1), axis=0)
        
        chart_kind = 'bar'
        title = 'Proportion of Hate Speech Categories Across Days of the Week'
        xlabel = 'Day of the Week'
        xticks = None
        rotation = 0
    else:
        raise ValueError("Invalid time_dimension. Choose from: 'hour', 'day'.")

    # Derive category order from target_mapping for the legend
    if class_order is not None:
        existing_cols = [col for col in class_order if col in proportions.columns]
        proportions = proportions[existing_cols]

    # Create figure and axes 
    fig, ax = plt.subplots(figsize=figsize)

    proportions.plot(
        kind=chart_kind, 
        stacked=True, 
        cmap='tab10', 
        alpha=0.85, 
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)

    if xticks is not None:
        ax.set_xticks(xticks)
        
    ax.tick_params(axis='x', rotation=rotation)

    # Legend outside the plot area
    ax.legend(title='Target Categories', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()

    return fig