import numpy as np
def OwnGlobalBinarise(img, thresh, maxval):
        '''
        This function takes in a numpy array image and
        returns a corresponding mask that is a global
        binarisation on it based on a given threshold
        and maxval. Any elements in the array that is
        greater than or equals to the given threshold
        will be assigned maxval, else zero.  
        Parameters
        ----------
        img : {numpy.ndarray}
            The image to perform binarisation on.
        thresh : {int or float}
            The global threshold for binarisation.
        maxval : {np.uint8}
            The value assigned to an element that is greater
            than or equals to `thresh`.        
        Returns
        -------
        binarised_img : {numpy.ndarray, dtype=np.uint8}
            A binarised image of {0, 1}.
        '''    
        binarised_img = np.zeros(img.shape, np.uint16)
        binarised_img[img >=thresh] = maxval
        # print(binarised_img)
        return binarised_img
def generate_semicircle(center_x, center_y, radius, stepsize=0.1):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    """        

    x = np.arange(center_x, center_x+radius+stepsize, stepsize)
    y = np.sqrt(radius**2 - x**2)
    # y =  x**2

    # since each x value has two corresponding y-values, duplicate x-axis.
    # [::-1] is required to have the correct order of elements for plt.plot. 
    x = np.concatenate([x,x[::-1]])

    # concatenate y and flipped y. 
    y = np.concatenate([y,-y[::-1]])

    return x, y + center_y        
def HorizontalFlip(mask):
    
    '''
    This function figures out how to flip (also entails whether
    or not to flip) a given mammogram and its mask. The correct
    orientation is the breast being on the left (i.e. facing
    right) and it being the right side up. i.e. When the
    mammogram is oriented correctly, the breast is expected to
    be found in the bottom left quadrant of the frame.
    
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The corresponding mask of the CC image to flip.

    Returns
    -------
    horizontal_flip : {boolean}
        True means need to flip horizontally,
        False means otherwise.
    '''
    
    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2
    
    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)
    
    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])
    top_sum = sum(row_sum[0:y_center])
    bottom_sum = sum(row_sum[y_center:-1])
    
    if left_sum < right_sum:
        horizontal_flip = True
    else:
        horizontal_flip = False
        
    return horizontal_flip        
def CropBorders(img):
    
    '''
    This function crops 1% from all four sides of the given
    image.
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.
        
    Returns
    -------
    cropped_img: {numpy.ndarray}
        The cropped image.
    '''
    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * 0.01)
    r_crop = int(ncols * (1 - 0.04))
    u_crop = int(nrows * 0.01)
    d_crop = int(nrows * (1 - 0.04))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    
    return cropped_img   
