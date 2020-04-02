from . import BaseReconstructor
from fracPy import reconstruction_object as ro

class ePIE(BaseReconstructor):
    def __init__(self, reconstruction_object: ro):
        self.obj = reconstruction_object

    def start_reconstruction(self):
        # actual reconstruction engine goes here
        pass







def ePIE(obj):
    """
    Reconstruct obj using the ePIE algorithm. More information can be found at

    ... insert link ..


    :param obj: Reconstruction object
    :return:
    """
    return obj