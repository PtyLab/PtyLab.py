import tqdm #import progressbar
import urllib.request

def download_with_progress(url, filename):
    """
    Download a file and show a progress bar while doing so.

    Parameters
    ----------
    url: URL of data
    filename: path to save it

    Returns
    -------
    None
    """
    pbar = None
    last_position = 0

    def show_progress(block_num, block_size, total_size):
        nonlocal  pbar
        nonlocal last_position
        if pbar is None:
            pbar = tqdm.tqdm(total=total_size, unit='Mb', unit_scale=1e-6 )#True, leave=False )#.ProgressBar(maxval=total_size)
        #pbar.start()


        downloaded = block_num * block_size
        update = downloaded - last_position
        last_position = downloaded
        if downloaded < total_size:
            #pbar.moveto(downloaded/total_size)
            pbar.update(update)

    urllib.request.urlretrieve(url, filename, show_progress)

# test it
if __name__ == '__main__':
    download_with_progress('https://figshare.com/ndownloader/files/38419391', 'test.hdf5')