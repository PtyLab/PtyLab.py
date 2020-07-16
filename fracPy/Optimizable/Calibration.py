from skimage.transform import AffineTransform, matrix_transform,\
                              SimilarityTransform, EuclideanTransform
from skimage.filters import gaussian
from skimage.measure import ransac
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

from sklearn.cluster import KMeans
from scipy.ndimage import map_coordinates
import traceback

import matplotlib.animation as animation
from fracPy.utils.utils import ifft2c, fft2c
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData


class IlluminationCalibration(Optimizable):
    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData):
        # These statements don't copy any data, they just keep a reference to the object
        self.optimizable = optimizable
        self.experimentalData = experimentalData
        # the following arrays need to be copied in memory for processing
        self.ptychogram = deepcopy(self.experimentalData.ptychogram)
        self.initialPositions = deepcopy(self.optimizable.positions)
        # POSITIONS MUST DEFINE THE CENTER OF THE SPATIAL FREQUENCY REGION
        self.initialPositions = self.initialPositions - self.experimentalData.Nd/2 
        
        # initialize search variables
        self.searchGridSize = 10
        self.gaussSigma = 2
        self.plot = False
        self.calibrateRadius = False
        self.brightfieldIndices = None
        self.fit_mode = "EuclideanTransform"
        self.apertRadiusPixel = self.experimentalData.entrancePupilDiameter/2 / self.experimentalData.dxp
        self.apertRadiusPixel_init = self.apertRadiusPixel
        # coherent transfer function
        self.CTF = np.abs(np.sqrt(self.experimentalData.Xp**2 + self.experimentalData.Yp**2) <= self.experimentalData.entrancePupilDiameter/2) * 1
        # optical transfer function = 2x CTF
        self.OTF = np.abs(np.sqrt(self.experimentalData.Xp**2 + self.experimentalData.Yp**2) <= self.experimentalData.entrancePupilDiameter) * 1

    
        # initialize_error_search_space
        self.angleRange_x_init = np.sin(np.mgrid[0:360:10]/180.*np.pi)
        self.angleRange_y_init =  np.cos(np.mgrid[0:360:10]/180.*np.pi)
        self.gridSearch_x_init = np.mgrid[-self.searchGridSize:self.searchGridSize+1]
        self.gridSearch_y_init = np.mgrid[-self.searchGridSize:self.searchGridSize+1]
        self.x_range = len(self.gridSearch_x_init)
        self.y_range = len(self.gridSearch_y_init)
        
           
    def findBrightfielIndices(self, ptychogram):
        """
        Use a threshold to separate brightfield images from darkfield images.
        Threshold is obtained by using a K-means clustering algorithm

        Parameters
        ----------
        ptychogram : 3D array
            image array containing experimental images.

        Returns
        -------
        brightfieldIndices : 1D array
            bool index array where 1s represent brightfield images.

        """
        # sum all the images together into a single 2D image     
        intensities = np.sum(ptychogram,(1,2))
    
        # kmeans clustering method to find 2 clusters
        cluster = np.array([np.arange(ptychogram.shape[0]), intensities]).T
        kmeans = KMeans(n_clusters=2).fit(cluster)
        clusters = kmeans.labels_
        # one cluster is brightfield other darkfield
        cluster1 = (clusters == 0)
        cluster2 = (clusters == 1)
              
        # if np.mean(intensities[cluster1]) > np.mean(intensities[cluster2]):
        if len(cluster1) < len(cluster2):
            brightfieldIndices = cluster1
        else:
            brightfieldIndices = cluster2
        return brightfieldIndices
    
    
    def convertToFourierSpace(self, ptychogram):
        """
        Convert brightfield image stack to the Fourier domain and also
        increase contrast based on all data analysis.

        Parameters
        ----------
        ptychogram : 3D array
            image array containing experimental images.

        Returns
        -------
        FT_ptychogram : 3D array
            contrast enhanced FFT(ptychogram).

        """
        FT_ptychogram =  np.abs(fft2c(ptychogram)) 
        # get the mean for normalization
        normalization = np.mean(FT_ptychogram,0)
        # get noise values outside 2x NA
        mean_outside_support = np.mean(normalization[self.OTF==0])
        # replace noise with a constant term
        normalization[normalization<(3*mean_outside_support)] = 3*mean_outside_support
        FT_ptychogram = np.abs(FT_ptychogram / normalization)
        return FT_ptychogram
    
    
    def generateCircularArcsVectorized(self, initialPositions):
        """
        Generate the circular arcs representing the coherent transfer function
        boundaries at various XY locations. This is required for a grid-search
        method to minimize the risk of gradient descent optimization getting
        stuck in local minima.
        
        Rather than returning circular perimeters for each XY grid location,
        they will be filtered such that each XY grid locations has a circular
        arc containing the same number of array entries. For this reason the 
        arrays will be filtered. The non-equal number of points is caused by
        the circular perimeter going out of bounds and resulting in a cricular 
        arc.
        
        The circular arc array will be a 5D array such that vectorized arrays
        can be used. This reduces readability, but also greatly improves 
        performance.
        
        The returned circular arc array will be a 5D array with entries:
        circularArcs = np.zeros([
        1D  multiple radius locations
        2D  meshgrid for each search grid locations
        1D  intensity values for each pixel along the circular radius
        1D  above value for each X and Y coordinates 
        
        Parameters
        ----------
        initialPositions : 2D array
            X,Y position values for the current image being analysed.

        Returns
        -------
        circularArcs : 5D array
            compact search grid array in incomprehensible format to use
            numpy vectorization for increased speed
        """        
        
        self.gridSearch_x = self.gridSearch_x_init.copy()
        self.gridSearch_y = self.gridSearch_y_init.copy()
        self.angleRange_x = self.angleRange_x_init.copy()
        self.angleRange_y = self.angleRange_y_init.copy()
        self.radiusScanRange = self.radiusScanRangeInit.copy()

            
        point_number = self.angleRange_x.shape[0]
        R_number = len(self.radiusScanRange)
        
        # convert radius search array into a 4D array
        self.radiusScanRange = np.reshape(self.radiusScanRange, [-1, 1, 1, 1])
        # convert angular search array into a 4D array
        self.angleRange_x = np.reshape(self.angleRange_x, [1, 1, 1, -1])
        self.angleRange_y = np.reshape(self.angleRange_y, [1, 1, 1, -1])
        # convert grid search mesh grid into a 4D array
        self.gridSearch_y, self.gridSearch_x = np.meshgrid(self.gridSearch_x, self.gridSearch_y)
        self.gridSearch_x = np.reshape(self.gridSearch_x, [1 ,self.x_range, self.y_range, 1])
        self.gridSearch_y = np.reshape(self.gridSearch_y, [1 ,self.x_range, self.y_range, 1])
    
        # compute the radial arc values for each spatial frequency position 
        # this is given in cartesian coordinates by:
        # x = Rsin(angle) + circle_center_x
        # y = Rcos(angle) + circle_center_y
        xx_circle_arc = np.single(self.radiusScanRange * self.angleRange_x\
                                  + self.experimentalData.Nd/2.\
                                  + initialPositions[0] \
                                  + self.gridSearch_x)
        yy_circle_arc = np.single(self.radiusScanRange * self.angleRange_y\
                                  + self.experimentalData.Nd/2.\
                                  + initialPositions[1]\
                                  + self.gridSearch_y)
         
        # convert back to a 1D array for filtering whether the circular arcs
        # are within the image boundaries
        xx_circle_arc = np.reshape(xx_circle_arc, [-1, point_number])
        yy_circle_arc = np.reshape(yy_circle_arc, [-1, point_number])
        
        # remove values such that each arc has the same number of elements
        outliers = ((yy_circle_arc > 1) *\
                      (xx_circle_arc > 1) *\
                      (xx_circle_arc < (self.experimentalData.Nd-1)) *\
                      (yy_circle_arc < (self.experimentalData.Nd-1))) == True    
        outliers = np.all(outliers,0)
        
        # the final circular arc array contains:
        # 1.   multiple radius locations (R_number)
        # 2-3. meshgrids for each search grid locations
        #      defined by (self.x_range, self.y_range)
        # 4.   intensity values for each pixel along the circular radius
        #      np.count_nonzero(outliers)
        # 5.   split the whole array into XY
        circularArcs = np.zeros([R_number, self.x_range, self.y_range, np.count_nonzero(outliers), 2]) 
              
        # generate circle arcs
        self.gridSearch_x = self.gridSearch_x_init.copy()
        self.gridSearch_y = self.gridSearch_y_init.copy()
        self.angleRange_x = self.angleRange_x_init.copy()
        self.angleRange_y = self.angleRange_y_init.copy()
        self.radiusScanRange = self.radiusScanRangeInit.copy()

        # convert radius search array into a 4D array
        self.radiusScanRange = np.reshape(self.radiusScanRange, [-1, 1, 1, 1])
        # convert angular search array into a 4D array
        self.angleRange_x = np.reshape(self.angleRange_x[outliers], [1, 1, 1, -1])
        self.angleRange_y = np.reshape(self.angleRange_y[outliers], [1, 1, 1, -1])
        # convert grid search mesh grid into a 4D array
        self.gridSearch_y, self.gridSearch_x = np.meshgrid(self.gridSearch_x, self.gridSearch_y)
        self.gridSearch_x = np.reshape(self.gridSearch_x, [1 ,self.x_range, self.y_range, 1])
        self.gridSearch_y = np.reshape(self.gridSearch_y, [1 ,self.x_range, self.y_range, 1])
    
        circularArcs[:, :, :, :, 0] = np.single(self.radiusScanRange * self.angleRange_x\
                                                 + self.experimentalData.Nd/2.\
                                                 + initialPositions[0]\
                                                 + self.gridSearch_x)
        circularArcs[:, :, :, :, 1] = np.single(self.radiusScanRange * self.angleRange_y\
                                                 + self.experimentalData.Nd/2.\
                                                 + initialPositions[1]\
                                                 + self.gridSearch_y)
        return circularArcs
    

    
    def calculateRMSE(self, image, error_candidates, position):
        """
        Calculate the RMSE between the experimental data and low-pass-filtered image,
        The low-pass filter is shifted based on calibrated coordinates. 
        Smallest RMSE gives the best k-space shift values

        Parameters
        ----------
        image : 2D array
            raw brightfield experimental image .
        error_candidates : 2D array
            grid search matrix containing the position error candidates.
        position : 2D array
            X,Y position values for the current image being analyses.

        Returns
        -------
        positions_error_x : float
            position error value x.
        positions_error_y : float
            position error value y.
        edge_case : bool
            this shows whether the error was at the edge of the search grid. If
            True then another search will be done using the updated positions.

        """        
        xx_img, yy_img = np.mgrid[0:self.experimentalData.Nd, 0:self.experimentalData.Nd]
        
        fft_raw = fft2c(image) 
        low_pass_img = image.copy()
        rms_array = []; x_arr = []; y_arr = []
        for error_candidate in error_candidates:
            positions_error_x = self.gridSearch_x_init[int(error_candidate[0])]
            positions_error_y = self.gridSearch_y_init[int(error_candidate[1])]
            
            # generate a shifted low-pass filter
            pupil = np.zeros(fft_raw.shape)
            pupil_coords = ((xx_img - self.experimentalData.Nd/2. + (position[0]+positions_error_x))**2 +\
                            (yy_img - self.experimentalData.Nd/2. + (position[1]+positions_error_y))**2) <= self.apertRadiusPixel**2
            pupil[pupil_coords] = 1
            # generate a low-pass filtered image with a shifted filter
            estimated_img = np.abs(fft2c(pupil * fft_raw))
            
            # see if the shifted positions minimize the error
            error = low_pass_img - estimated_img
            rms = np.sqrt(np.mean(error**2))
            rms_array.append(rms)
            x_arr.append(positions_error_x)
            y_arr.append(positions_error_y)
            
        err_idx = np.argmin(rms_array)
        positions_error_x = x_arr[err_idx]
        positions_error_y = y_arr[err_idx]
        
        # check if the value was on the edge of the error map
        b1 = 0
        b2 = len(self.gridSearch_x_init)
        if (error_candidates[err_idx,0] in [b1, b2]) or (error_candidates[err_idx,1] in [b1, b2]):
            edge_case = True
        else:
            edge_case = False
        return positions_error_x, positions_error_y, edge_case


    def fitCoordinates(self, src, dst, mode):
        """
        Parameters
        ----------
        src : 2D array
            source coordinates (to be calibrated)
        dst : 2D array
            destination coordinates (reference)
        mode : str
            Transformation mode: Translation, EuclideanTransform, SimilarityTransform, AffineTransform.

        Returns
        -------
        matrix : skimage transformation matrix
            3x3 transformation matrix.
        fitted : 2D array
            transformed source coordinates.

        """
        if mode == 'Translation' or mode == 'EuclideanTransform':
            tform_mode = EuclideanTransform
        elif mode == 'SimilarityTransform':
            tform_mode = SimilarityTransform
        elif mode == 'AffineTransform':
            tform_mode = AffineTransform
        else:
            tform_mode = EuclideanTransform
            print("Required fit mode not found, using EuclideanTransform")
            
        
        # residual threshold based on a median value
        resThresh = np.median(np.abs(dst-src))
        
        # compute the transformation matrix between the data points
        matrix, inliers = ransac((src, dst), tform_mode, min_samples=2,\
                                    residual_threshold=resThresh, max_trials=10000)

        if mode == 'Translation':
            matrix = tform_mode(translation=matrix.translation)   
        fitted = matrix_transform(src, matrix.params)
        return matrix, fitted

    
    def computePositionErrorCandidates(self, circleSearchArray):
        """
        Use the search grid error maps for each radial position and
        compute the gradients along the radial values. Highest gradient
        will indicate the best circular boundary location and these 
        grid search maps will be used as initial error candidates.
        1st and 2nd derivatives are used.
        
        
        Parameters
        ----------
        circleSearchArray : 3D array
            search grid error values for each radial position

        Returns
        -------
        errorCandidates : 2D array
            binary map indicating position error candidates.

        """
        # error map using first derivative
        # get the 1st derivative along the radial direction
        arc_dx1 = np.diff(circleSearchArray, n=1, axis=0)
        # get the largest radial gradient
        rad_dx_idx = np.argwhere(np.max(arc_dx1)==arc_dx1)[0][0]
        # get the search grid for the largest gradient
        search_grid_dx = np.squeeze(arc_dx1[rad_dx_idx,:,:])
        search_grid_dx=(search_grid_dx >= np.max(search_grid_dx) - 0.1*np.std(search_grid_dx)) # Select region around max

        # error map using second derivative
        # get the 2nd derivative along the radial direction
        arc_dx2 = np.diff(circleSearchArray, n=2, axis=0)
        # get the largest radial gradient
        rad_dx_dx_idx2 = np.argwhere(np.max(arc_dx2)==arc_dx2)[0][0]
        # get the search grid for the largest gradient
        search_grid_dx_dx = np.squeeze(arc_dx2[rad_dx_dx_idx2,:,:])
        search_grid_dx_dx=(search_grid_dx_dx >= np.max(search_grid_dx_dx) - 0.25*np.std(search_grid_dx_dx)) # Select region around max
        
        # combine the selected candidates
        errorCandidates = (search_grid_dx + search_grid_dx_dx)>0
        errorCandidates = np.argwhere(errorCandidates == True)
        return errorCandidates
    


    def findCalibratedRadius(self, ptychogram, FT_ptychogram, initialPositions):
        """
        Find the best aperture radius which corresponds to a wrong NA value.
        The actual radius is updated internally and is used for position 
        calibration.
        
        Parameters
        ----------
        ptychogram : 3D array
            image stack.
        FT_ptychogram : 3D array
            FFT(ptychogram).
        initialPositions : 2D array
            initial position vectors.

        Returns
        -------
        oldRadius : float
            old radius pre-calibration.
        newRadius : float
            new radius post-calibration.

        """
        self.radiuSearchBounds = 5
        self.radiusSearchStep = 0.5
        self.radiusSearchRange = np.mgrid[self.radiuSearchBounds:-self.radiuSearchBounds-self.radiusSearchStep:-self.radiusSearchStep]
        # radius tolerance array
        tolerance=[self.radiusSearchStep*1.1, self.radiusSearchStep*0.9]
        self.apertRadiusPixel_init = self.apertRadiusPixel
        
        converged = False
        check_done = False
        radii_array = [self.apertRadiusPixel]
        itr = 0
        while not converged:
            print("Radius {}px, iteration {}".format(self.apertRadiusPixel, itr))
            itr += 1
            # find the optimal radius for each image
            radiiIdxForEachImg = np.zeros(FT_ptychogram.shape[0])
            # radius used for testing
            self.radiusScanRangeInit = self.radiusSearchRange + self.apertRadiusPixel

            # use at most most 20 images
            idx_range = np.arange(FT_ptychogram.shape[0])
            if FT_ptychogram.shape[0] > 20:
                np.random.shuffle(idx_range)
                idx_range = idx_range[:20]
            
            for idx in idx_range:
                # compute circular arc used as candidates and make an error
                # storage array. This returns the radial arcs array
                radial_arcs = self.generateCircularArcsVectorized(initialPositions[idx,:])
                
                # flatten the coordinate array 
                coords = np.array([radial_arcs[:,:,:,:,0].flatten(),\
                                   radial_arcs[:,:,:,:,1].flatten()])
                # interpolate to avoid quantization errors
                image = gaussian(FT_ptychogram[idx,:,:], self.gaussSigma)     
                gridSearchForEachR = map_coordinates(image,\
                                                      coords,\
                                                      order=1,\
                                                      prefilter=False)
                
                # create the mean interpolated intensities along the perimeter for each
                # circular radius and grid search position
                circleSearchArray = np.mean(np.reshape(gridSearchForEachR,\
                                                         radial_arcs.shape[:-1]),3)
    
                # compute the gradient along the radial values
                arc_dx1 = np.gradient(circleSearchArray, axis=0)
                rad_dx_idx = np.argwhere(np.max(arc_dx1)==arc_dx1)[0][0]
                # best radius index for each image
                radiiIdxForEachImg[idx] = rad_dx_idx
    
            # find the median radius index
            radius_index = int(np.median(radiiIdxForEachImg))
            # select that radius corresponding to median
            self.apertRadiusPixel = self.radiusScanRangeInit[radius_index]
            radii_array.append(self.apertRadiusPixel)
        
            if not check_done:
                # check if radius change was small, then we are done
                if abs(radii_array[-1] - radii_array[-2]) <= tolerance[0]:
                    check_done = True
            else:
                # if the check was done
                mean_rad = np.mean(radii_array[-2:])
                if abs(mean_rad - radii_array[-1]) <= tolerance[1]:
                    # if within tolerance update the radius
                    self.apertRadiusPixel = mean_rad
                    converged = True  
                else:
                    # if deviation is too large then continue iterating
                    check_done = False
            
            if (itr >= 20 and not check_done) or self.apertRadiusPixel <= 0:
                self.apertRadiusPixel = self.apertRadiusPixel_init
                converged = True
               
        oldRadius = self.apertRadiusPixel_init
        newRadius = self.apertRadiusPixel
        return oldRadius, newRadius


    def findPositionCalibrationMatrix(self, ptychogram, FT_ptychogram, initialPositions):
        """
        Find the best aperture radius which corresponds to a wrong NA value.
        The actual radius is updated internally and is used for position 
        calibration.
        
        Parameters
        ----------
        ptychogram : 3D array
            image stack.
        FT_ptychogram : 3D array
            FFT(ptychogram).
        initialPositions : 2D array
            initial position vectors.

        Returns
        -------
        calibMatrix : skimage calibration matrix
            best transformation matrix that transforms initial coordinate grid
            to the updated positions.
        updatedPositions : 2D array
            updated positions, without the rigid transformation applied.

        """  
        calibrated_positions = initialPositions.copy()
        self.radiusSearchStep = 0.5 
        self.radiusSearchRange = np.mgrid[max(self.gaussSigma+2*self.radiusSearchStep,3*self.radiusSearchStep):-self.radiusSearchStep:-self.radiusSearchStep]
        self.radiusScanRangeInit = self.radiusSearchRange + self.apertRadiusPixel


        for idx in range(FT_ptychogram.shape[0]):
            image = gaussian(FT_ptychogram[idx,:,:], self.gaussSigma)     
            
            #######################################################################
            # repeat the fitting if the solution has not converged
            #######################################################################
            converged = False
            repeat_counter = 0
            try:
                # iterate to account for an error too far away from the initial
                # search grid 
                while not converged:
                    ##########################################################
                    # vectorized interpolation and mean value generation
                    ##########################################################
                    # compute circular arc used as candidates and make an error
                    # storage array. This returns the radial arcs array
                    radial_arcs = self.generateCircularArcsVectorized(calibrated_positions[idx])
                    
                    # flatten the coordinate array 
                    coords = np.array([radial_arcs[:,:,:,:,0].flatten(),\
                                       radial_arcs[:,:,:,:,1].flatten()])
                    # interpolate to avoid quantization errors
                    gridSearchForEachR = map_coordinates(image,\
                                                          coords,\
                                                          order=1,\
                                                          prefilter=False)
                    # define
                    circleSearchArray = np.mean(np.reshape(gridSearchForEachR,\
                                                             radial_arcs.shape[:-1]),3)
    
                    # compute the position error candidates
                    positionErrorCandidates = self.computePositionErrorCandidates(circleSearchArray)
                        
                    # go through all the possible coordinates and find the best one
                    positions_error_x, positions_error_y, values_on_edge =\
                        self.calculateRMSE(ptychogram[idx,:,:],\
                                        positionErrorCandidates,\
                                        calibrated_positions[idx])
    
                    # update coordinate grid
                    calibrated_positions[idx,0] = calibrated_positions[idx,0] + positions_error_x
                    calibrated_positions[idx,1] = calibrated_positions[idx,1] + positions_error_y                    
                    
                    # if the initial guess was too far and did not fall within 
                    # the initial search grid, try again
                    if values_on_edge and repeat_counter<5:
                        repeat_counter += 1
                    else:
                        converged = True
            except Exception:
                traceback.print_exc()
    
        # once finished fit the whole data
        calibMatrix, updatedPositions =\
            self.fitCoordinates(initialPositions, calibrated_positions, self.fit_mode)
        return calibMatrix, updatedPositions


    
    
    def plotCalibration(self, FT_ptychogram, initialPositions, calibrated_positions):
        """
        Plot the fitted circles and the calibrated positions pre and post 
        calibration

        Parameters
        ----------
        FT_ptychogram : 2D array
            FFT(ptychogram)
        initialPositions : 2D array
            initial positions pre calibration.
        calibrated_positions : 2D array
            positions post calibration.
        """
        
        fig = plt.figure(3)
        ax = fig.add_subplot(111)
        plt.title('position calibration results')
        ims = []
        
        angles_x = np.sin(np.mgrid[0:360:.1]/180.*np.pi)
        angles_y = np.cos(np.mgrid[0:360:.1]/180.*np.pi)
        for idx in range(FT_ptychogram.shape[0]):
            image = gaussian(FT_ptychogram[idx,:,:], self.gaussSigma)

            initial_row = self.apertRadiusPixel_init * angles_x +\
                self.experimentalData.Nd/2 + initialPositions[idx,0]
            initial_col = self.apertRadiusPixel_init * angles_y +\
                self.experimentalData.Nd/2 + initialPositions[idx,1]
    
            final_row = self.apertRadiusPixel * angles_x +\
                self.experimentalData.Nd/2 + calibrated_positions[idx,0]
            final_col = self.apertRadiusPixel * angles_y +\
                self.experimentalData.Nd/2 + calibrated_positions[idx,1]
            
            im1 = plt.imshow(image, animated=True)
            im2 = plt.scatter(initial_col, initial_row, label='initial guess',c='g',s=1)
            im3 = plt.scatter(final_col, final_row, label='calibrated', c='r',s=1)
            # produce a legend with the unique colors from the scatter
            legend = plt.legend(labels=['initial guess', 'calibrated'], loc="upper right")
            ims.append([im1,im2,im3,legend])
        
        # animate results
        animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)
        plt.show()
        
        # plot the scattered positions
        plt.figure(4)
        plt.scatter(initialPositions[:,1],initialPositions[:,0], label='initial guess',c='g',s=1)
        plt.scatter(calibrated_positions[:,1],calibrated_positions[:,0], label='calibrated', c='r',s=1)
        plt.grid(True)
        ax.set_aspect(aspect='equal')
        plt.legend()
        plt.show()
        return None
    
    
    def runCalibration(self):
        """
        Perform the calibration steps.
        1. Convert image array to Fourier space and enhacne contrast.
        2. Find brightfield indictes. Could be provided by the user or
        K-means clustering algorithm is used for automation.
        3. Calibrate NA/radius
        4. Calibrate positions
        
        Returns
        -------
        positionsFitted : 2D array
            positions after applying the correction matrix.
        entrancePupilDiameter : float
            updated aperture diameter.
        calibMatrix : skimage transformation matrix
            calibration matrix.

        """
        # get the FFT(ptychogram) which will also be filtered to enhance
        # contrast
        FFT_ptychogram = self.convertToFourierSpace(self.ptychogram)
        
        if not hasattr(self.brightfieldIndices, '__len__'):
            # find the brightfield indices
            self.brightfieldIndices = self.findBrightfielIndices(FFT_ptychogram)
            print("Number of brightfield images", np.count_nonzero(self.brightfieldIndices == 1))
            
        # calibrate the aperture radius / NA
        if self.calibrateRadius:
            oldRadius, newRadius = self.findCalibratedRadius(self.ptychogram[self.brightfieldIndices],
                                                            FFT_ptychogram[self.brightfieldIndices],
                                                            self.initialPositions[self.brightfieldIndices])
            print("Initial radius was {}px".format(np.round(oldRadius,2)))
            print("Calibrated radius is {}px".format(np.round(newRadius,2)))
            oldNA = np.round(oldRadius / self.experimentalData.dxp * self.experimentalData.wavelength,2)
            newNA = np.round(newRadius / self.experimentalData.dxp * self.experimentalData.wavelength,2)
            print("Initial NA was {}".format(oldNA))
            print("Calibrated NA is {}".format(newNA))


        # find the calibration matrix between the initial positions and the ones
        # found based on circle fitting
        calibMatrix, updatedPositions = self.findPositionCalibrationMatrix(self.ptychogram[self.brightfieldIndices],
                                                                             FFT_ptychogram[self.brightfieldIndices],
                                                                             self.initialPositions[self.brightfieldIndices])
        # fit the positions
        positionsFitted = matrix_transform(self.initialPositions, calibMatrix.params)
        
        # plot the results
        if self.plot:
            self.plotCalibration(FFT_ptychogram, self.initialPositions, positionsFitted)

        # shift the positions back to where they were initially
        positionsFitted = positionsFitted + self.experimentalData.Nd/2 
        
        # update the entrancePupilDiameter
        entrancePupilDiameter = self.apertRadiusPixel * self.experimentalData.dxp * 2

        return positionsFitted, entrancePupilDiameter, calibMatrix
