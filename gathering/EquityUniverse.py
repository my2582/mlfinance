import pandas as pd
from datetime import date
from utils.misc import get_excel_column_name, get_last_bday

class EquityUniverse(object):
    """Loads a broader universe and screens out with filtering conditions.
    """

    def __init__(self, csv_name, min_market_cap=200):
        """Sets a path at which a csv-format broader universe file exists.

        Parameters
        ----------
        csv_name: string
            A csv file name from which a raw universe file is loaded.

        min_market_cap: int
            The minimum market cap in USD million.
        """
        self.universe = pd.read_csv(csv_name,
                                header=1,
                                names=[
                                    'Identifier', 'Name', 'Revenue', 'Company Type',
                                    'Business Description', 'FactSet Industry',
                                    'Crunchbase Category(BETA)',
                                    'Crunchbase Rank(BETA)', 'Ultimate Parent Name',
                                    'Fiscal Year End', 'Country', 'Website'
                                ], 
                                parse_dates = ['Fiscal Year End']
                                )
        
        # Change dtypes properly.
        self.universe.loc[:, 'Revenue'] = pd.to_numeric(self.universe.loc[:,'Revenue'], errors='coerce')
        self.universe.loc[:, 'Fiscal Year End'] = pd.to_datetime(self.universe.loc[:, 'Fiscal Year End'], errors='coerce')

        # Set the minimum market cap.
        self.min_market_cap = min_market_cap

        # Drop rows not satisfying the minimum.
        self.universe = self.universe.loc[self.universe.loc[:, 'Revenue'] >= min_market_cap, :]
        self.universe = self.universe.reset_index(drop = True)

    def save(self, name='./dataset/universe/feather/eqy_universe_gt50mm.feather', pickle=True):
        """Saves `universe` into a feather, optionally, and pickle formatted file.

        Arguments:
        ----------
        pickle: boolean
            Pickle it if True. We feather it as a default.
        """
        self.universe.to_feather(name)
        if pickle:
            name = name.split('.')[:-1] + ['pkl']
            name = '.'.join(name)
            self.universe.to_pickle(name)

    def get_universe(self):
        """
        Returns:
        --------
        universe: pd.DataFrame
        """
        return self.universe
