def remove_night_time_data(df):
    """
    Remove night time entries in df
    
    Input
    -----
        df : pandas dataframe
    
    Output
    ------
        df : pandas dataframe
    """
    night_power = -1
    df          = df[df.Power != night_power]
    return df
