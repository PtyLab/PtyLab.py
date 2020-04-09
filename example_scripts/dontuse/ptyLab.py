import pickle
import logging

logging.basicConfig(level=logging.DEBUG)

class DataLoader:
    """
    This is a container class for all the data associated with the ptychography reconstruction.

    It only holds attributes that are the same for every type of reconstruction.

    Things that belong to a particular type of reconstructor are stored in the .params class of that particular reconstruction.

    """

    def __init__(self, datafolder):
        self.logger = logging.getLogger('PtyLab')
        self.logger.debug('Initializing PtyLab object')
        self.dataFolder = datafolder
        self.initialize_attributes()

        #self.prepare_reconstruction()



    def prepare_visualisation(self):
        """ Create figure and axes for visual feedback.

        """
        return NotImplementedError()






    def save(self, name='obj'):
        with open(self.dataFolder.joinpath('%s.pkl' % name), 'wb') as openfile:
            pickle.dump(self, openfile)

    def load_from_hdf5(self):
        """ Load preprocessed data from an hdf5 file."""
        raise NotImplementedError()

    def load(self, name='obj'):
        raise NotImplementedError()

    def transfer_to_gpu_if_applicable(self):
        """ Implements checkGPU"""
        pass



    ### Functions that still have to be implemented
    def checkDataset(self):
        raise NotImplementedError()

    def checkFFT(self):
        raise NotImplementedError()

    def getOverlap(self):
        """
        :return:
        """
        raise NotImplementedError()

    def initialParams(self):
        """ Initialize the params object and attach it to the object

        Note that this is a little bit different from the matlab implementation
        """
        self.params = Params()

    def setPositionOrder(self):
        raise NotImplementedError()

    def showDiffractionData(self):
        raise NotImplementedError()

    def showPtychogram(self):
        raise NotImplementedError()

    ## Special properties
    # so far none

    def prepare_reconstruction(self):
        """
        Prepare the reconstruction. So far, followin matlab, it checks the following things:

        - minimize memory footprint
        -
        :return:
        """
        # delete parts of the memory that are not required.
        self.checkMemory()
        # do something with the modes
        self.checkModes
        # prepare FFTs
        self.checkFFT()
        # prepare vis
        self.prepare_visualisation()
        # This should not be necessary
        # obj.checkMISC
        # transfer to GPU if req'd
        self.checkGPU()

    # Things we'd like to change the name of
    def checkGPU(self, *args, **kwargs):
        return self.transfer_to_gpu_if_applicable(*args, **kwargs)





if __name__ == '__main__':
    obj = DataLoader('.')
