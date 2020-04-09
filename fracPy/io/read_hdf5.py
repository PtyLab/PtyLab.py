

def check_data_fields(filename):
    """
    Make sure that all the fields in a given .hdf5 file are supported and do some sanity checks.

    This is run before loading the file just to make sure that the file is correctly formatted.
    :param filename: '.hdf5' file with all the necessary attributes.
    :return: None if correct
    :raise: KeyError if one of the attributes is missing.
    """
    # Tomas: Please implement this, I think it should go along these lines:

    required_keys = ['I', 'dont', 'know']
    archive = None # load_archive(filename) # This is your part
    # Feel free to change it but this is the gist
    for k in required_keys:
        if k not in archive.keys():
            raise KeyError('hdf5 file misses key %s' % k)


