"""
class to format the input data
"""
import ast
import pandas as pd


class InputParser:
    """
    class to format the input data
    """

    df: pd.DataFrame = pd.DataFrame()

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.parse()

    def get(self) -> pd.DataFrame:
        """
        :return: the parsed DataFrame with converted types ready to be computed
        """
        return self.df

    def rename(self) -> None:
        """
        rename input data to be computed correctly
        """
        self.df.rename(
            columns={"Pressure (Bar)": "Pressure", "Temperature (K)": "Temperature"},
            inplace=True,
        )

    @staticmethod
    def column_names() -> tuple[str, str, str]:
        """
        function to return some dataset column names
        :return: a tuple containing the most useful names of the dataset columns
        """
        phi = "Phi"
        temperature = "Temperature"
        pressure = "Pressure"
        return phi, temperature, pressure

    def fix_unit_measures(self):
        """
        add unit measures for Temperature, Pressure
        """
        self.df["Temperature Symbol"] = "K"
        self.df["Pressure Symbol"] = "Bar"

    def fix_column_types(self):
        """
        converts dataframe types to the correct ones
        """
        phi, temp, press = self.column_names()
        self.df[phi] = self.df[phi].apply(ast.literal_eval)
        self.df[temp] = self.df[temp].apply(ast.literal_eval)
        self.df[press] = self.df[press].apply(ast.literal_eval)

    def split_tuples_into_columns(self, columns: tuple) -> None:
        """
        splits column of tuple into N columns where N is the len of the tuple

        E.g.
        input : columns = {a} with a containing (b,c)
        will create column a0 and a1 containing b and c respectively
        :param columns: a tuple containing the names of the column containing a tuple to split
        """
        added_columns = []
        for column in columns:
            new_column_names = [column + str(i) for i in range(len(self.df[column][0]))]
            added_columns += new_column_names
            for index, new_col in enumerate(new_column_names):
                self.df[new_col] = self.df[column].apply(
                    lambda _tuple, i=index: _tuple[i]
                )
        print(f"created columns {list(self.df.loc[1, added_columns].keys())}")

    def parse(self, df: pd.DataFrame = None) -> None:
        """
        parses the data
        :type df: pd.DataFrame
        :param df: to add the dataframe at runtime
        """
        if df:
            self.df = df

        if self.df.empty:
            return

        self.rename()
        self.fix_unit_measures()
        self.fix_column_types()
        self.split_tuples_into_columns(self.column_names())


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    format the input data
    :param df: pd.Dataframe
    """
    return InputParser(df).get()


if __name__ == "__main__":
    input_data = pd.read_excel(
        "/home/aron/Documents/Progetto_Data_mining/"
        "data-validation-of-fuel-combustion/input_files/data.xlsx"
    )
    InputParser(input_data)
