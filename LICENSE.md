Academic License Agreement
================================

Please note that Phase Focus Limited, UK, holds a portfolio of international patents regarding ptychography, which you can find listed here: https://www.phasefocus.com/patents
Implementations of ptychography included within this PTYLAB computer software program fall within the scope of patents owned by Phase Focus Limited.
If you intend to pursue ANY commercial interest in technologies making use of the patents, please contact Phase Focus Limited to discuss a commercial-use licence here: https://www.phasefocus.com/licence

********************************
This license agreement sets forth the terms and conditions under which the authors (hereafter "LICENSOR") grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for academic, non-commercial purposes ONLY (hereafter "LICENSE") to use the PTYLAB ptychography computer software program and associated documentation furnished hereunder (hereafter "PROGRAM").
Terms and Conditions of the LICENSE
 1.	LICENSOR grants to LICENSEE a royalty-free, non-exclusive license to use the PROGRAM for academic, non-commercial purposes, upon the terms and conditions hereinafter set out and until termination of this license as set forth below.
 2.	LICENSEE acknowledges that the PROGRAM is a research tool still in the development stage. The PROGRAM is provided without any related services, improvements or warranties from LICENSOR and that the LICENSE is entered into in order to enable others to utilize the PROGRAM in their academic activities. It is the LICENSEE's responsibility to ensure its proper use and the correctness of the results.
 3.	THE PROGRAM IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT OF ANY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS. IN NO EVENT SHALL THE LICENSOR, THE AUTHORS OR THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY ARISING FROM, OUT OF OR IN CONNECTION WITH THE PROGRAM OR THE USE OF THE PROGRAM OR OTHER DEALINGS IN THE PROGRAM.
 4.	LICENSEE agrees that it will use the PROGRAM and any modifications, improvements, or derivatives of PROGRAM that LICENSEE may create (collectively, "IMPROVEMENTS") solely for academic, non-commercial purposes and that any copy of PROGRAM or derivatives thereof shall be distributed only under the same license as PROGRAM. The terms "academic, non-commercial", as used in this Agreement, mean academic or other scholarly research which (a) is not undertaken for profit, or (b) is not intended to produce works, services, or data for commercial use, or (c) is neither conducted, nor funded, by a person or an entity engaged in the commercial use, application or exploitation of works similar to the PROGRAM.
 5.	LICENSEE agrees that it shall make the following acknowledgement in any publication resulting from the use of the PROGRAM or any translation of the code into another computing language:
       "Loetgering, Lars, et al. 'ptyLab: a cross-platform inverse modeling toolbox for conventional and Fourier ptychography.' Computational Optical Sensing and Imaging. Optical Society of America, 2021."
 6.	Except for the above-mentioned acknowledgment, LICENSEE shall not use the PROGRAM title or the names or logos of LICENSOR, nor any adaptation thereof, nor the names of any of its employees or laboratories, in any advertising, promotional or sales material without prior written consent obtained from LICENSOR in each case.
 7.	Ownership of all rights, including copyright in the PROGRAM and in any material associated therewith, shall at all times remain with LICENSOR, and LICENSEE agrees to preserve same. LICENSEE agrees not to use any portion of the PROGRAM or of any IMPROVEMENTS in any machine-readable form outside the PROGRAM, nor to make any copies except for its internal use, without prior written consent of LICENSOR. LICENSEE agrees to place this licence on any such copies.
 8.	The LICENSE shall not be construed to confer any rights upon LICENSEE by implication or otherwise except as specifically set forth herein.
 9.	This Agreement shall be governed by the material laws of UNITED KINGDOM and any dispute arising out of this Agreement or use of the PROGRAM shall be brought before the courts of England.
********************************
Further recommendations on how to cite this code and prior art:
-- ptyLab
Loetgering, Lars, et al.
"ptyLab: a cross-platform inverse modeling toolbox for conventional and Fourier ptychography."
Computational Optical Sensing and Imaging. Optical Society of America, 2021.
********************************

Specific algorithms:
- conventional ptychography -
- ePIE
Maiden, Andrew M., and John M. Rodenburg.
"An improved ptychographical phase retrieval algorithm for diffractive imaging."
Ultramicroscopy 109.10 (2009): 1256-1262.
- mPIE
Maiden, Andrew, Daniel Johnson, and Peng Li.
"Further improvements to the ptychographical iterative engine." Optica 4.7 (2017): 736-745.
- pcPIE
Zhang, Fucai, et al.
"Translation position determination in ptychographic coherent diffraction imaging."
Optics express 21.11 (2013): 13592-13606.
- e3PIE
Maiden, Andrew M., Martin J. Humphry, and John M. Rodenburg.
"Ptychographic transmission microscopy in three dimensions using a multi-slice approach."
JOSA A 29.8 (2012): 1606-1614.
- lsqPIE
Please cite ePIE + the following reference
Odstrčil, Michal, Andreas Menzel, and Manuel Guizar-Sicairos.
"Iterative least-squares solver for generalized maximum-likelihood ptychography."
Optics express 26.3 (2018): 3108-3123.
- aPIE
de Beurs, Anne, et al. "aPIE: an angle calibration algorithm for reflection ptychography." Optics Letters 47.8 (2022): 1949-1952.
- zPIE
Loetgering, Lars, et al.
"zPIE: an autofocusing algorithm for ptychography."
Optics letters 45.7 (2020): 2030-2033.
- pSD + Poisson intensity update
Thibault, P., and M. Guizar-Sicairos.
"Maximum-likelihood refinement for coherent diffractive imaging."
New Journal of Physics 14.6 (2012): 063004.
- mixed states
Thibault, Pierre, and Andreas Menzel.
"Reconstructing state mixtures from diffraction measurements."
Nature 494.7435 (2013): 68-71.
- multispectral ptychography
Batey, Darren J., Daniel Claus, and John M. Rodenburg.
"Information multiplexing in ptychography."
Ultramicroscopy 138 (2014): 13-21.
- orthogonal probe relaxation
Odstrcil, Michal, et al.
"Ptychographic coherent diffractive imaging with orthogonal probe relaxation."
Optics express 24.8 (2016): 8360-8369.
- mqNewton = PIE + mPIE
Rodenburg, John M., and Helen ML Faulkner.
"A phase retrieval algorithm for shifting illumination."
Applied physics letters 85.20 (2004): 4795-4797.
(also cite mPIE here)
- ptychographic OCT + external reference beam recovery (interferometric update)
Du, Mengqi, et al.
"Ptychographic optical coherence tomography."
Optics Letters 46.6 (2021): 1337-1340.
- Fourier ptychography -
- Basic Fourier ptychography implementation (with known pupil via illumination scanning)
Zheng, Guoan, Roarke Horstmeyer, and Changhuei Yang. "Wide-field, high-resolution Fourier ptychographic microscopy." Nature photonics 7.9 (2013): 739-745.
- Basic Fourier ptychography implementation (with known pupil and pupil scanning)
Dong, Siyuan, et al. "Aperture-scanning Fourier ptychography for 3D refocusing and super-resolution macroscopic imaging." Optics express 22.11 (2014): 13586-13599.
- recovery of pupil function
Ou, Xiaoze, Guoan Zheng, and Changhuei Yang. "Embedded pupil function recovery for Fourier ptychographic microscopy." Optics express 22.5 (2014): 4960-4972.
- estimation of illumination directions
Eckert, Regina, Zachary F. Phillips, and Laura Waller. "Efficient illumination angle self-calibration in Fourier ptychography." Applied optics 57.19 (2018): 5434-5442.