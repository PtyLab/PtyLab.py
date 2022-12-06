from PtyLab.utils.utils import circ, gaussian2D, cart2pol
from PtyLab.utils.scanGrids import GenerateNonUniformFermat
from PtyLab.Operators.Operators import aspw
from PtyLab.utils.visualisation import hsvplot, show3Dslider
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import convolve2d


class PtySim:
    def __init__(self, param):
        self.wavelength = param["wavelength"]

        # detector coordinates
        self.z0 = param["z0"]
        self.Nd = param["Nd"]
        self.dxd = param["dxd"]
        self.Ld = self.Nd * self.dxd

        # probe coordinates
        self.dxp = self.wavelength * self.z0 / self.Ld
        self.Np = self.Nd
        self.Lp = self.dxp * self.Np
        self.xp = np.arange(-self.Np // 2, self.Np // 2) * self.dxp
        self.Xp, self.Yp = np.meshgrid(self.xp, self.xp)

        # object coordinates
        self.No = param["No"]
        self.dxo = self.dxp
        self.Lo = self.dxo * self.No
        self.xo = np.arange(-self.No // 2, self.No // 2) * self.dxo
        self.Xo, self.Yo = np.meshgrid(self.xo, self.xo)

        # Creates the folders where the data sould be saved
        # utils.create_dir(self.savedir)
        # utils.create_dir(self.savedir_plots)
        # utils.create_dir(self.savedir_dp)

    def init_probe(self, probe_type, **kwargs):

        probe_type = probe_type.lower()

        if probe_type == "simulate_focus":
            # Simulate a focused beam
            f = kwargs["f"]
            pinhole = circ(self.Xp, self.Yp, self.Lp / 2)
            pinhole = convolve2d(
                pinhole, gaussian2D(5, 1).astype(np.float32), mode="same"
            )

            probe = aspw(pinhole, 2 * f, self.wavelength, self.Lp)[0]

            aperture = circ(self.Xp, self.Yp, 3 * self.Lp / 4)
            aperture = convolve2d(
                aperture, gaussian2D(5, 3).astype(np.float32), mode="same"
            )
            probe = (
                probe
                * np.exp(
                    -1.0j
                    * 2
                    * np.pi
                    / self.wavelength
                    * (self.Xp**2 + self.Yp**2)
                    / (2 * f)
                )
                * aperture
            )
            self.probe = aspw(probe, 2 * f, self.wavelength, self.Lp)[0]

        elif probe_type == "mat":
            # Load probe to simulate from a mat file
            if "p_filepath" in kwargs and "file_psize" in kwargs:
                probe = loadmat(kwargs["p_filepath"])["probe"][0]
                probe_psize = kwargs["file_psize"]
                factor = probe_psize / self.psize_real_space

                # Adjust for the different pixel size
                temp_r = zoom(np.real(probe), factor, order=0)
                temp_i = zoom(np.imag(probe), factor, order=0)
                probe = temp_r + 1j * temp_i

                if not probe.shape[0] == self.d_shape:
                    probe = utils.array_utils.crop_center(
                        probe, self.d_shape, self.d_shape
                    )

        plt.figure(figsize=(5, 5), num=1)
        ax1 = plt.subplot(121)
        hsvplot(self.probe, ax=ax1, pixelSize=self.dxp)
        ax1.set_title("complex probe")
        plt.subplot(122)
        plt.imshow(abs(self.probe) ** 2)
        plt.title("probe intensity")
        plt.show(block=False)

    def init_obj(self, obj_type, **kwargs):
        """
        # Docstring has to be written...
        """

        if obj_type == "image":
            pass
            # # Load the sample as .png
            # sample = imread(filepath_sample)[:, :, 0] / 255
            # # sample = np.round(sample / 255)
            # sample[sample >= 0.99] = 1
            # sample[sample < 0.99] = 0
            #
            # plt.figure()
            # plt.imshow(sample)
            #
            # zoom_factor = self.file_psize / self.psize_real_space
            #
            # print(zoom_factor)
            #
            # sample = zoom(sample, zoom_factor, order=0)
            #
            # plt.figure()
            # plt.imshow(sample)

        elif obj_type == "polychrom_image":
            pass
            # """
            # Image with different grey values that correspond to different materials
            # """
            # # Load the sample as .png
            # sample = imread(filepath_sample)[:, :]
            # sample[sample == 0] = 0  # absorbing
            # sample[sample == 64] = 1  # Silizium
            # sample[sample == 144] = 2  # Al
            # sample[sample == 254] = 3  # Vac
            #
            # zoom_factor = self.file_psize / self.psize_real_space
            # sample = zoom(sample, zoom_factor, order=0)

        elif obj_type == "spiral":
            # Generate a spiral pattern as object

            d = kwargs["d"]
            b = kwargs["b"]

            # d = 1e-3  # the smaller this parameter the larger the spatial frequencies in the simulated object
            # b = 33  # topological charge (feel free to play with this number)
            theta, rho = cart2pol(self.Xo, self.Yo)
            t = (1 + np.sign(np.sin(b * theta + 2 * np.pi * (rho / d) ** 2))) / 2
            phaseFun = 1
            t = t * circ(self.Xo, self.Yo, self.Lo) * (
                1 - circ(self.Xo, self.Yo, 200 * self.dxo)
            ) * phaseFun + circ(self.Xo, self.Yo, 130 * self.dxo)
            obj = convolve2d(t, gaussian2D(5, 3), mode="same")  # smooth edges
            # obj_phase = np.exp(1.j*2*np.pi/wavelength*(Xo**2+Yo**2)*20)
            self.object = obj * phaseFun

        plt.figure(figsize=(5, 5), num=2)
        ax = plt.axes()
        hsvplot(np.squeeze(self.object), ax=ax)
        ax.set_title("Complex Object")
        plt.show()

    def init_coordinates(
        self, pos_spacing, pos_extent, pos_model="spiral", plot_scan_pattern=True
    ):
        pass

    def init_spectrum(self, N_spec, **kwargs):
        pass

    def calc_diff(self, **kwargs):
        pass

    def add_detector(self, **kwargs):
        pass

    def plot_diff_data(self):
        pass

    def export_data(self, filename):
        with h5py.File(fileName + ".hdf5", "w") as hf:
            hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
            hf.create_dataset("encoder", data=encoder, dtype="f")
            hf.create_dataset("binningFactor", data=binningFactor, dtype="i")
            hf.create_dataset("dxd", data=(dxd,), dtype="f")
            hf.create_dataset("Nd", data=(Nd,), dtype="i")
            hf.create_dataset("No", data=(No,), dtype="i")
            hf.create_dataset("zo", data=(zo,), dtype="f")
            hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
            hf.create_dataset(
                "entrancePupilDiameter", data=(entrancePupilDiameter,), dtype="f"
            )
            hf.close()
            print("An hd5f file has been saved")

        pass


if __name__ == "__main__":

    phys_param = {
        "wavelength": 632.8e-9,
        "z0": 50e-3,
        "Nd": 2**7,
        "dxd": 2**11 / 2**7 * 4.5e-6,
        "No": 2**10 + 2**9,
    }

    vis_sim = PtySim(phys_param)
    vis_sim.init_probe(probe_type="simulate_focus", f=5e-3)
    vis_sim.init_obj(obj_type="spiral", d=1e-3, b=33)
