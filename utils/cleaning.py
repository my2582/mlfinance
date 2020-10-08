import numpy as np
import pandas as pd
import re
import itertools
from typing import List


class DataCleaner:
    def __init__(self, logger = None):
        self.logger = logger

    def log(self, *args, logtype='debug', sep=' '):
        assert self.logger is not None, 'self.logger is not initialized yet at instantiation'
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def remove_empty(self, text: List[str]):
        '''
        Returns a new list of `text` where any empty string '' is removed.

        Parameters:
        text : List[str]
            a list of string possibly including '' to be removed.
        '''
        return list(filter(None, text))

    def splitter(self, n_splits: int, text: str, sep = ' - '):
        '''
        Returns a list of strings separated by `sep` in `text`, but only first `n_splits`.
        The remaining parts will be the last element of a returned-list.
        e.g. text = ['Don - CEO - Director']
        splitter(2, text)
        >> ['Don', 'CEO - Director']

        Parameters:
        n_splits: integer
            The number of words splitted with `sep`.

        text: str
            A string possibly including `sep`

        sep: str or List
            A separator to split `text`
            e.g. sep = [' - ', ' = '] or str = ' - '
        '''
        # If multiple column separators are passed in List type,
        # a separator splitting a row the most will be chosen through `best_sep_idx`.
        if isinstance(sep, List):
            best_sep_idx = np.argmax([len(self._splitter(n_splits=n_splits, text=text, sep=sep)) for s in sep])
            best_sep = sep[best_sep_idx]
            return self._splitter(n_splits, text, sep=best_sep)
        else:
            return self._splitter(n_splits, text, sep=sep)
    
    def _splitter(self, n_splits: int, text: str, sep = ' - '):
        '''
        Returns a list of strings separated by `sep` in `text`, but only first `n_splits`.
        The remaining parts will be the last element of a returned-list.
        Functions as the same as splitter(), but `sep` only accepts string type.
        e.g. text = ['Don - CEO - Director']
        splitter(2, text)
        >> ['Don', 'CEO - Director']

        Parameters:
        n_splits: integer
            The number of words splitted with `sep`.

        text: str
            A string possibly including `sep`

        sep: str 
            A separator to split `text`
            e.g. str = ' - '
        '''
        s_text = text.split(sep)
        n_splits = min(len(s_text), n_splits)
        result = [s_text[i] for i in range(n_splits-1)]
        remaining_text = s_text[n_splits-1:]
        result.append(sep.join(remaining_text))

        return result

    def convert_multilines(self, text: str, col_nm=['col1', 'col2'], ret_type='df', col_sep=' - ', row_sep='\n'):
        '''
        Returns either a DataFrame or dictionary instance where columns are extracted by splitting with `col_sep` and each row is separated by `row_sep`.
        Any empty row will be dropped. `col_sep` may have multiple separators, separated by '|'. e.g.: col_sep = ' - | = '. In this case, a separator splitting a row the most will be chosen.
        
        e.g.
            text = '\n\nSunny Sanyal - President - Chief Executive Officer\n\n\nSam Maheshwari\nClarence Verhoef - Retiring Chief Financial Officer\nHoward Goldman - Director of Investor Relations\n'
            convert_multilines(text, col_nm=['name', 'title'], ret_type='dic')
            >> {'name': {0: 'Sunny Sanyal',
                1: 'Sam Maheshwari',
                2: 'Clarence Verhoef',
                3: 'Howard Goldman'},
                'title': {0: 'President - Chief Executive Officer',
                1: None,
                2: 'Retiring Chief Financial Officer',
                3: 'Director of Investor Relations'}}

        Parameters:
        text: str
            A string consisting of multiple lines where each line is separated by `row_sep`

        col_nm: list
            A list of column names. Default is ['col1', 'col2']

        ret_type: {'df'|'dic}
            Either 'df' or 'dic', which indicates a return type. A DataFrame instance is returned if 'df', else a dictionary instance.
            
        col_sep: str or List
            A string of column separators, possibly containing multiple ones.
            e.g. col_sep = ' - ' or col_sep = [' - ', ' = ']
            
        row_sep: str
            A row separator to split `text` into multiple rows if needed.
        '''
        
        # In case multiple column separators are passed, set `multiple_col_seps` to be True.
        multiple_col_seps = True if isinstance(col_sep, List) else False

        # Turn `text` into multiple rows by splitting with `row_sep`.
        rows = text.split(row_sep)
        rows = self.remove_empty(rows)
        col_size = len(col_nm)
        
        # Do the column-splitting.
        records = []
        for row in rows:
            # If multiple column separators are passed,
            # a separator splitting a row the most will be chosen through `best_sep_idx`.
            if multiple_col_seps:
                best_sep_idx = np.argmax([len(self.splitter(n_splits=col_size, text=row, sep=sep)) for sep in col_sep])
                best_col_sep = col_sep[best_sep_idx]
            else:
                best_col_sep = col_sep
                
            col = self.splitter(col_size, row, sep=best_col_sep)
            records.append(col)

        return pd.DataFrame(records, columns=col_nm) if ret_type == 'df' else pd.DataFrame(records, columns=col_nm).to_dict()

    def get_between_strings(self, begin_with: str, end_with: str, text: str, sep='|'):
        '''
        Returns a text between two strings. None is returned if no match. The first match is returned if multiple matches.
        e.g. Matching criteria would be: "Find the text that comes after any element of ['Executives|Company Participants|Company Representatives'] and that comes before any element of ['Analysts|Conference Call Participants']."
        
        Parameters:
        begin_with : str
            String of beginning words separated by `sep` within that string. e.g. ['Executives|Company Participants|Company Representatives']
        
        end_with: str
            String of ending words separated by `sep` within that string. e.g. ['Analysts|Conference Call Participants']
        
        text: str
            a text body to be searched over.
        
        sep: str
            Both `begin_with` and `end_with` will by splitted by `sep` and all combinations of separated words are used to do the search.
        '''
        begin_str = begin_with.split(sep)
        end_str = end_with.split(sep)
        
        # Compile regular expresions for different combinations of `begin_str` and `end_str`.
        matches = [[re.search(r'(?<=({}))(.|\n)+(?={})'.format(b,s), string=text) for b in begin_str] for s in end_str]
        
        # Flatten a list of lists
        matches =list(itertools.chain.from_iterable(matches))
        
        # Iterate over the matched results and return the first matched.
        for match in matches:
            if match is not None:
                self.log('Matched:')
                return match.group()
        
        return None

    def get_regexp(self, type='ticker'):
        '''
        Returns a compiled regular expression for `type`

        Parameters:
        type: str
            This may be one of {'ticker'}
    
        Returns:
            If the value of `type` is
            'ticker': returns a regular expression for uppercase alphanumeric letters between ( and )
        '''
        if type == 'ticker':
            return re.compile(r'(?<=\()[A-Z0-9.-]+(?=\))')