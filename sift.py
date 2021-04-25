import cv2
import numpy as np

def computeKeypointsAndDescriptors(image):
    image = image.astype('float32')
    print('Generate base image...')
    base_image = generateBaseImage(image)
    print('Generate Gaussian images...')
    gaussian_images = generateGaussianImages(base_image)
    print('Generate DoG images...')
    dog_images = generateDoGImages(gaussian_images)
    print('Generate keypoints...')
    keypoints = generateKeypoints(gaussian_images, dog_images)
    print('Generate discriptors...')
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors

def generateBaseImage(image):
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    baseImage = cv2.GaussianBlur(image, (0, 0), sigmaX=1.25, sigmaY=1.25)  
    return baseImage

def generateGaussianImages(image):
    gaussian_images = []
    sigma = np.array([1.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    num_octaves = int(np.log(min(image.shape))/np.log(2)-1)
    
    for image_index in range(0, 5):
        sigma[image_index+1] = 1.2265*(1.26 ** image_index)
    
    for octave_index in range(num_octaves):
        octave_images = [image]
        for s in sigma[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=s, sigmaY=s)
            octave_images.append(image)
        gaussian_images.append(octave_images)
        octave_base = octave_images[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1]/2), int(octave_base.shape[0]/2)))
    
    return np.array(gaussian_images)

def generateDoGImages(gaussian_images):
    dog_images = []

    for octave_images in gaussian_images:
        dog_octave_images = []
        for i in range(len(octave_images)-1):
            dog_octave_images.append(cv2.subtract(octave_images[i], octave_images[i+1]))
        dog_images.append(dog_octave_images)
    
    return np.array(dog_images)

def generateKeypoints(gaussian_images, dog_images):
    keypoints = []

    for octave_index, dog_octave_images in enumerate(dog_images):
        for idx in range(len(dog_octave_images)-2):
            for i in range(5, dog_octave_images[idx].shape[0]-5):
                for j in range(5, dog_octave_images[idx].shape[1]-5):
                    keypoint, image_index = localizeKeypoint(idx+1, i, j, octave_index, dog_octave_images)
                    if keypoint is not None:
                        new_keypoints = computeAngle(keypoint, octave_index, gaussian_images[octave_index][image_index])
                        keypoints += new_keypoints
    
    remove_list = []
    last = keypoints[0]
    
    for keypoint in keypoints[1:]:
        if last.pt == keypoint.pt and last.size == keypoint.size and last.angle == keypoint.angle:
            remove_list.append(keypoint)
        last = keypoint

    for keypoint in remove_list:
        keypoints.remove(keypoint)
            
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5*np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave-1) & 255)
    
    return keypoints

def localizeKeypoint(idx, i, j, octave_index, dog_octave_images):
    image_cube = np.stack([dog_octave_images[idx-1][i-1:i+2, j-1:j+2], dog_octave_images[idx][i-1:i+2, j-1:j+2], dog_octave_images[idx+1][i-1:i+2, j-1:j+2]])
    
    isExtremum = False
    if abs(image_cube[1, 1, 1]) > 1:
        if image_cube[1, 1, 1] > 0:
            isExtremum = np.all(image_cube[1, 1, 1] >= image_cube)
        elif image_cube[1, 1, 1] < 0:
            isExtremum = np.all(image_cube[1, 1, 1] <= image_cube)
    if not isExtremum:
        return None, None

    image_shape = dog_octave_images[0].shape
    
    for t in range(4):
        image1, image2, image3 = dog_octave_images[idx-1][i-1:i+2, j-1:j+2], dog_octave_images[idx][i-1:i+2, j-1:j+2], dog_octave_images[idx+1][i-1:i+2, j-1:j+2]
        gradient = computeGradient(image1, image2, image3)
        hessian = computeHessian(image1, image2, image3)
        update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.all(abs(update) < 0.5):
            break
        j += int(np.round(update[0]))
        i += int(np.round(update[1]))
        idx += int(np.round(update[2]))
        if i < 5 or i >= image_shape[0]-5 or j < 5 or j >= image_shape[1]-5 or idx < 1 or idx > 3:
            return None, None
    
    newExtremum = image2[1][1] + 0.5*np.dot(gradient, update)
    if abs(newExtremum) >= 0.0:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and xy_hessian_trace < 3.4785*np.sqrt(xy_hessian_det):
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j+update[0])*(2 ** octave_index), (i+update[1])*(2 ** octave_index))
            keypoint.octave = octave_index+idx*(2 ** 8)+int((update[2]+0.5)*255)*(2 ** 16)
            keypoint.size = 3.2*(2 ** ((idx+update[2])/3+octave_index))
            keypoint.response = abs(newExtremum)
            return keypoint, idx
    
    return None, None

def computeGradient(image1, image2, image3):
    array = np.stack([image1, image2, image3]).astype('float32')/255
    dx = 0.5*(array[1, 1, 2]-array[1, 1, 0])
    dy = 0.5*(array[1, 2, 1]-array[1, 0, 1])
    dz = 0.5*(array[2, 1, 1]-array[0, 1, 1])
    return np.array([dx, dy, dz])

def computeHessian(image1, image2, image3):
    array = np.stack([image1, image2, image3]).astype('float32')/255
    dxx = array[1, 1, 2]-2*array[1, 1, 1]+array[1, 1, 0]
    dyy = array[1, 2, 1]-2*array[1, 1, 1]+array[1, 0, 1]
    dzz = array[2, 1, 1]-2*array[1, 1, 1]+array[0, 1, 1]
    dxy = 0.25*(array[1, 2, 2]-array[1, 2, 0]-array[1, 0, 2]+array[1, 0, 0])
    dyz = 0.25*(array[2, 2, 1]-array[2, 0, 1]-array[0, 2, 1]+array[0, 0, 1])
    dxz = 0.25*(array[2, 1, 2]-array[2, 1, 0]-array[0, 1, 2]+array[0, 1, 0])
    return np.array([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])

def computeAngle(keypoint, octave_index, gaussian_image):
    new_keypoints = []
    image_shape = gaussian_image.shape

    scale = 0.75*keypoint.size/(2**octave_index)
    radius = int(3*scale)
    raw_histogram = np.zeros(36)
    smooth_histogram = np.zeros(36)

    for i in range(-radius, radius+1):
        region_y = int(np.round(keypoint.pt[1]/(2 ** octave_index)))+i
        if region_y > 0 and region_y < image_shape[0]-1:
            for j in range(-radius, radius + 1):
                region_x = int(np.round(keypoint.pt[0]/(2 ** octave_index)))+j
                if region_x > 0 and region_x < image_shape[1]-1:
                    dx = gaussian_image[region_y, region_x+1]-gaussian_image[region_y, region_x-1]
                    dy = gaussian_image[region_y-1, region_x]-gaussian_image[region_y+1, region_x]
                    histogram_index = int(np.round(np.rad2deg(np.arctan2(dy, dx))/10))
                    raw_histogram[histogram_index%36] += np.exp(-0.5*(i ** 2 + j ** 2)/(scale**2))*np.sqrt(dx ** 2 + dy ** 2)

    for n in range(36):
        smooth_histogram[n] = (6*raw_histogram[n]+4*(raw_histogram[n-1]+raw_histogram[(n+1)%36])+raw_histogram[n-2]+raw_histogram[(n+2)%36])/16
    
    angle_max = max(smooth_histogram)
    angle_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    
    for peak_index in angle_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= 0.8*angle_max:
            new_peak_index = (peak_index+0.5*(smooth_histogram[(peak_index-1)%36]-smooth_histogram[(peak_index+1)%36])/(smooth_histogram[(peak_index-1)%36]-2*peak_value+smooth_histogram[(peak_index+1)%36]))%36
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, 360-new_peak_index*10, keypoint.response, keypoint.octave)
            new_keypoints.append(new_keypoint)
    
    return new_keypoints

def generateDescriptors(keypoints, gaussian_images):
    descriptors = []
    for keypoint in keypoints:
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1/np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        
        gaussian_image = gaussian_images[octave+1, layer]
        num_rows, num_cols = gaussian_image.shape
        x, y = np.round(scale*np.array(keypoint.pt)).astype('int')
        cos = np.cos(np.deg2rad(360-keypoint.angle))
        sin = np.sin(np.deg2rad(360-keypoint.angle))
        row_list = []
        col_list = []
        magnitude_list = []
        orientation_list = []
        histogram = np.zeros((6, 6, 8))

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = 1.5*scale*keypoint.size 
        half_width = int(min(3.54*hist_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width+1):
            for col in range(-half_width, half_width+1):
                row_bin = ((col*sin+row*cos)/hist_width)
                col_bin = ((col*cos-row*sin)/hist_width)
                if abs(row_bin) < 2.5 and abs(col_bin) < 2.5:
                    window_row = y+row
                    window_col = x+col
                    if window_row > 0 and window_row < num_rows-1 and window_col > 0 and window_col < num_cols-1:
                        dx = gaussian_image[window_row, window_col+1]-gaussian_image[window_row, window_col-1]
                        dy = gaussian_image[window_row-1, window_col]-gaussian_image[window_row+1, window_col]
                        gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))%360
                        row_list.append((row_bin-1.5))
                        col_list.append((col_bin-1.5))
                        magnitude_list.append(np.exp(-0.125*(row_bin ** 2 + col_bin ** 2))*gradient_magnitude)
                        orientation_list.append((keypoint.angle+gradient_orientation-360)/45)
        
        for row, col, magnitude, orientation in zip(row_list, col_list, magnitude_list, orientation_list):
            # Smoothing via trilinear interpolation
            row_floor, col_floor, orientation_floor = np.floor([row, col, orientation]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row-row_floor, col-col_floor, orientation-orientation_floor
            #if orientation_floor < 0:
            #    orientation_floor += 8
            #if orientation_floor >= 8:
            #    orientation_floor -= 8

            c1 = magnitude*row_fraction
            c0 = magnitude*(1-row_fraction)
            c11 = c1*col_fraction
            c10 = c1*(1-col_fraction)
            c01 = c0*col_fraction
            c00 = c0*(1-col_fraction)
            c111 = c11*orientation_fraction
            c110 = c11*(1-orientation_fraction)
            c101 = c10*orientation_fraction
            c100 = c10*(1-orientation_fraction)
            c011 = c01*orientation_fraction
            c010 = c01*(1-orientation_fraction)
            c001 = c00*orientation_fraction
            c000 = c00*(1-orientation_fraction)

            histogram[row_floor+1, col_floor+1, orientation_floor] += c000
            histogram[row_floor+1, col_floor+1, (orientation_floor+1)%8] += c001
            histogram[row_floor+1, col_floor+2, orientation_floor] += c010
            histogram[row_floor+1, col_floor+2, (orientation_floor+1)%8] += c011
            histogram[row_floor+2, col_floor+1, orientation_floor] += c100
            histogram[row_floor+2, col_floor+1, (orientation_floor+1)%8] += c101
            histogram[row_floor+2, col_floor+2, orientation_floor] += c110
            histogram[row_floor+2, col_floor+2, (orientation_floor+1)%8] += c111
        
        descriptor_vector = histogram[1:-1, 1:-1, :].flatten()
        # Threshold and normalize descriptor_vector
        threshold = 0.2*np.linalg.norm(descriptor_vector)
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512*descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')
