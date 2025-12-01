import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler

def preprocess_fitness(df: pd.DataFrame, seprator: str = "_", dateEmbedding: bool = False) -> pd.DataFrame:
    """
    Process dataset for columns, date (embedding with sine/cosine), unique_log_id (unique id for a record)
    Args: 
        df (DataFrame): pandas dataframe from a dataset
        seprator (str): a string to concatinating participant_id, activity_type and date string
        dateEnbedding (bool): if date convert into sine and cosine embedding
    Returns:
        DataFrame: processed dataframe 
    """
    df.drop(columns=['gender'], errors='ignore', inplace = True)

    ## unique_log_id : concating 3 columns (participant_id, activity_type, date)
    df['unique_log_id'] = (
        df['participant_id'].astype(str) + seprator +
        df['activity_type'].astype(str) + seprator +
        df['date'].astype(str)
    )

    ## date convert
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(by=['participant_id', 'date'], ascending=True)
    if dateEmbedding: 
        # extract date attributes
        date = df['date'].dt
        month = date.month.values
        weekday = date.weekday.values
        day = date.day.values
        quarter = date.quarter.values
        dayofyear = date.dayofyear.values

        ## embedding date to sin/cos
        df['month_sin'], df['month_cos'] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
        df['weekday_sin'], df['weekday_cos'] = np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)
        df['day_sin'], df['day_cos'] = np.sin(2 * np.pi * day / 31), np.cos(2 * np.pi * day / 31)
        df['quarter_sin'], df['quarter_cos'] = np.sin(2 * np.pi * quarter / 4), np.cos(2 * np.pi * quarter / 4)
        df['dayOfYear_sin'], df['dayOfYear_cos'] = np.sin(2*np.pi* dayofyear / 365), np.cos(2*np.pi* dayofyear / 365)

        ## drop date column
        df.drop(columns=['date'], inplace=True)

    # convert to category
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].astype('category')

    return df


def standardize_fitness(fitness: pd.DataFrame) -> pd.DataFrame:
    """
    stanadarized numerical columns
    Args: 
        df (DataFrame): pandas dataframe from a dataset
    Returns: 
        (DataFrame): processed dataframe
    """
    cols_to_scale = [
        'age', 'height_cm', 'weight_kg', 'bmi', 'duration_minutes',
        'daily_steps', 'avg_heart_rate', 'resting_heart_rate',
        'sleep_hours', 'hydration_level', 'endurance_level',
        'blood_pressure_diastolic', 'blood_pressure_systolic', 'calories_burned'
    ]
    existing_cols = [c for c in cols_to_scale if c in fitness.columns]

    scaler = StandardScaler(copy=False)  # 避免額外複製
    fitness[existing_cols] = scaler.fit_transform(fitness[existing_cols])

    return fitness


def activities_based(
    df: pd.DataFrame,
    exclude: Union[str, list] = None, 
    activity: Union[str, list] = None,
    onehot: bool = False,
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Filtered dataset with specific activities
    Args: 
        df (DataFrame): pandas dataframe from a dataset
        exclude (Union[str, list]): columns will be excluded for one hot encoded
        activity (Union[str, list]): filtered specific activities
        onehot (bool): one hot encode to columns with category type
        drop_original (bool): remain original dataframe
    Returns: 
        (DataFrame): processed dataframe
    """
    if activity is not None:
        if isinstance(activity, str):
            mask = df['activity_type'] == activity
        elif isinstance(activity, list):
            mask = df['activity_type'].isin(activity)
        else:
            raise ValueError("activity must be str or list[str]")
        df = df.loc[mask].reset_index(drop=True)

    if 'activity_type' in df.columns:
        df = df.drop(columns=['activity_type'])

    if isinstance(exclude, str):
        exclude = [exclude]
    elif exclude is None:
        exclude = []

    # One-hot encoding
    if onehot:
        cat_cols = df.select_dtypes(include=['category', 'object']).columns.difference(exclude)
        if len(cat_cols) > 0:
            if drop_original:
                df = pd.get_dummies(df, columns=cat_cols, dtype=np.uint8)
            else:
                onehot_df = pd.get_dummies(df[cat_cols], dtype=np.uint8)
                df = pd.concat([df, onehot_df], axis=1)

    return df
