import pandas as pd


def save_to_csv(results, params, filename, **kwargs):
    """
    Takes a vector of results, a list of parameters, and optional keyword arguments and compiles them into a pandas
    DataFrame which is in turn saved to disk as as csv.

    Arguments:
        results (list): list of results of equal length to params

        params (list): list of parameter values and names

        filename (str): location to save the csv to

        **kwargs: optional keyword arguments are entered into every row of the DataFrame
    """
    # Create empty pandas DataFrame
    data = pd.DataFrame()
    # Tidy up params
    params = tidy_params(params)
    # Check to make sure that params and results are the same length
    if len(params) != len(results):
        raise ValueError('params and results should have the same length!')
    # Loop through results and params
    for result, paramdict in zip(results, params):
        # Filter out _input from paramdict
        paramdict.pop('_input')
        # Append result to paramdict
        paramdict['result'] = result
        # Go through all kwargs and add to row
        for key, value in kwargs.items():
            paramdict[key] = value
        # Append to dataframe
        data = data.append(paramdict, ignore_index=True)
    # Save to disk
    data.to_csv(filename)


def tidy_params(params):
    """
    Transforms a possibly nested list of parameters encoded as dicts into a flat list. Each element of the input that is
    a list and not a dict is processed recursively until we find the first element of the most nested list and this is
    returned.

    Arguments:
        params (list): possibly nested list of dicts. If it's just a list of dicts, it will be returned as is, otherwise
            the nested lists will be unnested and their component dicts will be combined

    Returns:
        output: flattened params list
    """
    # Define function to handle logic
    def inner(_element):
        # If it's a list, process the first element recursively
        if type(_element) is list:
            return inner(_element[0])
        # If we have a dict, just return nothing
        elif type(_element) is dict:
            return _element
        else:
            raise TypeError('tidy_params received an unexpected type in params')

    # Create empty list to store output
    output = list()
    # Loop through elements of tidy_params
    for element in params:
        output.append(inner(element))

    return output
