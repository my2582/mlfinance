import pandas as pd

def get_excel_column_name(i, letter='A'):
    """Returns the alphabet that is `i`-th apart from `letter`, which is compatible with the column name in Excel.
    e.g.: get_letter(26) returns 'AA'
    
    Parameters
    ----------
    letter : character
        The letter from which `i`-th apart.
    
    i : int
        `i`-th apart from `letter.
    """
    if i <= 25:
        return chr(ord(letter)+i)
    else:
        return get_excel_column_name((i // 26)-1) + get_excel_column_name(i % 26)


def get_last_bday(date_from=None, holidays=[''], weekmask='Mon Tue Wed Thu Fri'):
    """Returns the last business day from `date_from`.

    Parameters
    ----------
    date_from : string ('%Y-%m-%d')

    holidays : list
        list/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``

    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        weekmask of valid business days, passed to ``numpy.busdaycalendar``

    Returns
    -------
    string
        The last business day
    """
    bday = pd.tseries.offsets.CustomBusinessDay(holidays=holidays, weekmask=weekmask)
    if date_from is None:
        date_from = pd.Timestamp.today()

    return (date_from - bday).strftime('%Y-%m-%d')