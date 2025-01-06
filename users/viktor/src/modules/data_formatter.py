import pandas as pd
from interfaces import IDataFormatter

class DataFormatter(IDataFormatter):
    """
    A class for formatting and transforming a DataFrame by renaming specific columns
    and adding new columns where applicable. Implements IDataFormatter.
    """

    def __init__(self, df: pd.DataFrame, column_renames: dict, special_handlings_columns: dict):
        """
        Initializes the DataFormatter with a DataFrame.

        Args:
            df (pd.DataFrame): The dataset to be formatted.
            column_renames (dict): Dictionary mapping old column names to new names.
            special_handlings_columns (dict): Dictionary of columns that require special handling.
        """
        super().__init__(df, column_renames, special_handlings_columns)

    def rename_columns(self) -> pd.DataFrame:
        """
        Renames specific columns in the DataFrame and creates new columns if applicable:
            - 'Id' is renamed to 'job_id'.
            - 'Description' is renamed to 'description'.
            - 'Title' is renamed to 'title'.
            - 'Language' is renamed to 'language'.
            - 'CreatedAt' is renamed to 'original_listed_time'.
            - 'Compensation' and 'med_salary' are both renamed to 'med_salary_monthly'.
            - 'AddressId' is renamed to 'location'. Additionally, a duplicate column named 'company_name' 
              is created from the renamed 'location' column.

        The method checks for the existence of each column before attempting renaming, ensuring 
        compatibility with DataFrames that may not contain all the specified columns.

        Returns:
            pd.DataFrame: The modified DataFrame with renamed columns and any additional columns added.
        """
        rename_mapping = {}

        # Iterate through the column_renames dictionary to check and apply renaming
        for old_name, new_name in self.column_renames.items():
            if old_name in self.df.columns:
                rename_mapping[old_name] = new_name
            else:
                # Log or handle missing columns, for example by raising a warning or skipping
                print(f"Warning: Column '{old_name}' not found in the DataFrame.")

        # Apply renaming if there are any valid mappings
        if rename_mapping:
            self.df = self.df.rename(columns=rename_mapping)
        
        # Special handling for creating additional columns from specific ones
        for old_name, new_column in self.special_handlings_columns.items():
            if old_name in self.df.columns:
                self.df[new_column] = self.df[old_name]
            else:
                # Handle missing columns for special processing
                print(f"Warning: Column '{old_name}' for special handling not found in the DataFrame.")

        return self.df