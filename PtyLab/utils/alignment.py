import numpy as np
from PtyLab import Reconstruction, ExperimentalData, Params, Engines


def show_alignment(
    reconstruction: Reconstruction,
    data: ExperimentalData,
    params: Params,
    engine: Engines.BaseEngine,
):
    """Show a viewer which gives information about the alignment of the probe and the object"""
    import napari

    viewer = napari.Viewer()
    # sort all the images by distance to the center
    mean_pos = reconstruction.positions.mean(0, keepdims=True)
    order = np.argsort(np.linalg.norm(reconstruction.positions - mean_pos, axis=-1))[
        :15
    ]
    ptycho_ordered = data.ptychogram[order]
    # do one iteration of all of them

    from PtyLab.Operators.Operators import object2detector, detector2object

    reconstruction.initializeObjectProbe()
    reconstruction.esw = reconstruction.probe
    # do all the propagators
    from PtyLab.Operators.Operators import fresnelPropagator

    z0_0 = reconstruction.zo
    # esw, updated_esw = detector2object(ptycho_ordered-ptycho_ordered.mean(), params, reconstruction)
    # updated_esw = updated_esw[0,0,0]
    # print(updated_esw.shape)
    # viewer.add_image(abs(updated_esw**2), name='refocused')
    # move them so they 'overlap'
    # for i, position in enumerate(reconstruction.positions[order]-reconstruction.positions[order][0]):
    #     # do one iteration for all of them and keep them in memory
    #     row, col = position
    #     updated_esw[i] = np.roll(np.roll(updated_esw[i], axis=-2, shift=col),
    #                              axis=-1, shift=row)
    # viewer.add_image(abs(updated_esw**2), name='aligned refocused')

    viewer.add_image(ptycho_ordered, name="ptychogram radially ordered")
    # propagate them
    viewer.show()
