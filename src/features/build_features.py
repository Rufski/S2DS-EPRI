import numpy as np

def find_true_cleaning_events(df, inplace=False):
    """ Find actual cleaning events from soiling factor.
    """
    soiling_factor = df.Soiling.to_numpy()
    first_value = soiling_factor[0]

    if inplace:
        df['soiling_loss'] = np.concatenate((np.array([first_value - 1]), soiling_factor[1:]-soiling_factor[:-1]))
        df['cleaning_event'] = (df['soiling_loss'] > 0).astype(int)

        return df
    else:
        output_df = pd.DataFrame(index=df.index)
        output_df['soiling_loss'] = np.concatenate((np.array([first_value - 1]), soiling_factor[1:]-soiling_factor[:-1]))
        output_df['cleaning_event'] = (output_df['soiling_loss'] > 0).astype(int)
        return output_df
