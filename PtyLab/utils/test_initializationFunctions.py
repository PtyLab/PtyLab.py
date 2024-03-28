from unittest import TestCase
import PtyLab


class TestInitialProbeOrObject(TestCase):
    def setUp(self) -> None:
        experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
            'example:helicalbeam', operationMode="CPM", engine=PtyLab.Engines.mPIE
        )

        self.experimentalData = experimentalData
        self.reconstruction = reconstruction

        self.params = params
        self.monitor = monitor
        self.ePIE = ePIE_engine
        self.experimentalData.setOrientation(4)

    def test_initial_probe_or_object(self):
        print(self.experimentalData.entrancePupilDiameter / self.reconstruction.dxo)
        # make a tiny probe

        self.experimentalData.entrancePupilDiameter = 30 * self.reconstruction.dxo

        self.reconstruction = PtyLab.Reconstruction(self.experimentalData, self.params)
        self.reconstruction.copyAttributesFromExperiment(self.experimentalData)

        self.reconstruction.initialProbe = 'circ_smooth'

        self.reconstruction.initializeObjectProbe()

        self.ePIE = PtyLab.Engines.mPIE(self.reconstruction, self.experimentalData,
                                        params=self.params, monitor=self.monitor)

        print(self.reconstruction.entrancePupilDiameter)
        self.ePIE.numIterations = 50
        self.monitor.probeZoom = None  # show full FOV

        self.ePIE.reconstruct()
