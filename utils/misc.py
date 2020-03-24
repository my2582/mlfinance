import pandas as pd

def get_letter(i, letter='A'):
    """Returns the alphabet that is `i`-th apart from `letter`.
    
    Parameters
    ----------
    letter : character
        The letter from which `i`-th apart.
    
    i : int
        `i`-th apart from `letter.
    """
    return chr(ord(letter) + i)


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