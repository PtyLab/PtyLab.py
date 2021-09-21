import numpy as np
from fracPy import Reconstruction, ExperimentalData, Params, Engines

def show_alignment(reconstruction: Reconstruction, data: ExperimentalData, params: Params,
                   engine: Engines.BaseEngine):
    """ Show a viewer which gives information about the alignment of the probe and the object """
    import napari
    viewer = napari.Viewer()
    # sort all the images by distance to the center
    mean_pos = reconstruction.positions.mean(0, keepdims=True)
    order = np.argsort(np.linalg.norm(reconstruction.positions - mean_pos, axis=-1))
    ptycho_ordered = data.ptychogram[order]

    engine.detector2object(esw=ptycho_ordered)
    viewer.add_image(abs(reconstruction.esw**2), name='backfocused')

    viewer.add_image(ptycho_ordered, name='ptychogram radially ordered')
    # propagate them
    viewer.show()

