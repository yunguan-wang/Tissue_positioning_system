import os
import re
import sys
import math
import numbers
import numpy as np
import skimage
import skimage.transform

CHANNEL_AXIS = -1
SKIMAGE_VERSION = skimage.__version__


def unpack_dict(kwargs, N):
    """ Unpack a dictionary of values into a list (N) of dictionaries. """
    return [dict((k, v[i]) for k, v in kwargs.items()) for i in range(N)]


def unique_colors(x, channel_axis=None):
    if not channel_axis:
        return np.unique(x)
    else:
        return np.unique(x.reshape(channel_axis, x.shape[channel_axis]), axis=0)


def image_stats(x, channel_axis=None):
    if x is None:
        return None
    stats = [x.min(), x.max(), len(unique_colors(x, channel_axis))] if min(x.shape) > 0 else [None, None, None]
    return [x.shape, x.dtype] + stats


def img_as(dtype):
    """ Convert images between different data types. 
        (Note that: skimage.convert is not a public function. )
        If input image has the same dtype and range, function will do nothing.
        (This check is included in skimage.convert, so no need to implement it here. )
        https://github.com/scikit-image/scikit-image/blob/master/skimage/util/dtype.py
        dtype: a string or a python dtype or numpy.dtype: 
               'float', 'float32', 'float64', 'uint8', 'int32', 'int64', 'bool', 
               float, uint8, bool, int,
               np.floating, np.float32, np.uint8, np.int, np.bool, etc
    """
    dtype = np.dtype(dtype)
    # return lambda x: skimage.convert(x, dtype, force_copy=False)
    dtype_name = dtype.name
    if dtype_name.startswith('float'):
        # convert(image, np.floating, force_copy=False)
        if dtype_name == 'float32':
            return skimage.img_as_float32
        elif dtype_name == 'float64':
            return skimage.img_as_float64
        else:
            return skimage.img_as_float
    elif dtype_name == 'uint8':
        # convert(image, np.uint8, force_copy=False)
        return skimage.img_as_ubyte
    elif dtype_name.startswith('uint'):
        # convert(image, np.uint16, force_copy=False)
        return skimage.img_as_uint
    elif dtype_name.startswith('int'):
        # convert(image, np.int16, force_copy=False)
        return skimage.img_as_int
    elif dtype_name == 'bool':
        # convert(image, np.bool_, force_copy)
        return skimage.img_as_bool
    else:
        raise ValueError(f"{dtype_name} is not a supported data type in skimage.")


def resize(img, size, order=1, **kwargs):
    """ Resize the input numpy array image to the given size.
    Args:
        img (numpy array): Image to be resized.
        size (tuple): Desired output size. 
        order (int, optional): Desired interpolation. Default is 1
        kwargs: other parameters for skimage.transform.resize
    Returns:
        numpy array: Resized image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be numpy array. Got {}'.format(type(img)))
    
    mode = kwargs.setdefault('mode', 'reflect')
    cval = kwargs.setdefault('cval', 0.)
    clip = kwargs.setdefault('clip', True)
    preserve_range = kwargs.setdefault('preserve_range', False)
    args = ({'anti_aliasing': kwargs.setdefault('anti_aliasing', True),
             'anti_aliasing_sigma': kwargs.setdefault('anti_aliasing_sigma', None)} 
            if SKIMAGE_VERSION > '0.14' else {})
    return skimage.transform.resize(img, output_shape=size, order=order, mode=mode, cval=cval, 
                                    clip=clip, preserve_range=preserve_range, **args)


class Resize(object):
    def __init__(self, size, order=1, **kwargs):
        self.size = size
        self.order = order
        self.kwargs = kwargs
    
    def __call__(self, images, kwargs=None):
        return [resize(_, self.size, self.order, **self.kwargs)
                if _ is not None else None
                for _ in images]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, order={1})'.format(self.size, self.order)


def get_pad_width(input_size, output_size, pos='center', stride=1):
    output_size = output_size + input_size[len(output_size):]
    output_size = np.maximum(input_size, output_size)
    if pos == 'center':
        l = np.floor_divide(output_size - input_size, 2)
    elif pos == 'random':
        # l = [np.random.randint(0, _ + 1) for _ in output_size - input_size]
        l = [np.random.randint(0, _ + 1) * stride for _ in (output_size - input_size)//stride]
    return list(zip(l, output_size - input_size - l))


def pad(img, size=None, pad_width=None, pos='center', mode='constant', **kwargs):
    """ Pad the input numpy array image with pad_width and to given size.
    Args:
        img (numpy array): Image to be resized.
        size (tuple): Desired output size. 
        pad_width (list of tuples): Desired pad_width. 
        pos: one of {'center, 'random'}, default is 'center'. if given
             size, the parameter will decide whether to put original 
             image in the center or a random location.
        mode: supported mode in skimage.util.pad
        kwargs: other parameters in skimage.util.pad
    
    pad_width and size can have same length as img, or 1d less than img.
    pad_width and size cannot be both None. If size = None, function will
    image with return img_size + pad_width. If pad_width = None, function 
    will return image with size. If both size and pad_width is not None,
    function will pad with pad_width first, then will try to meet size. 
    Function don't do any resize, rescale, crop process. Return img size 
    will be max(img.size+pad_width, size). 
    Returns:
        numpy array: Resized image.
    """
    if mode == 'constant':
        pars = {'constant_values': kwargs.setdefault('cval', 0.0)}
    elif mode == 'linear_ramp':
        pars = {'end_values': kwargs.setdefault('end_values', 0.0)}
    elif mode == 'reflect' or mode == 'symmetric':
        pars = {'reflect_type': kwargs.setdefault('reflect_type', 'even')}
    else:
        pars = {'stat_length': kwargs.setdefault('stat_length', None)}
    
    if pad_width is not None:
        pad_width = pad_width + [(0, 0)] * (img.ndim - len(pad_width))
        # img = skimage.util.pad(img, pad_width[:img.ndim], mode=mode, **pars)
        img = np.pad(img, pad_width[:img.ndim], mode=mode, **pars)
    
    if size is not None:
        pad_var = get_pad_width(img.shape, output_size=size, pos=pos)
        # img = skimage.util.pad(img, pad_var, mode=mode, **pars)
        img = np.pad(img, pad_var, mode=mode, **pars)
    
    return img


class Pad(object):
    def __init__(self, size=None, pad_width=None, pos='center', mode='constant', **kwargs):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        else:
            if size is not None:
                assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        
        self.size = size
        self.pad_width = pad_width
        self.pos = pos
        self.mode = mode
        self.kwargs = kwargs
    
    def __call__(self, images, kwargs=None):
        ## Add support to kwargs, let kwargs over-write self.kwargs for different inputs.
        ## like different cvals for different type of images.
        if kwargs is None:
            kwargs = [{}] * len(images)
        if self.pad_width is not None:
            images = [pad(img, size=None, pad_width=self.pad_width, 
                          mode=self.mode, **{**self.kwargs, **args}) 
                      if img is not None else None
                      for img, args in zip(images, kwargs)]
        if self.size is not None:
            pad_width = get_pad_width(images[0].shape, output_size=self.size, pos=self.pos)
            images = [pad(img, size=None, pad_width=pad_width, 
                          mode=self.mode, **{**self.kwargs, **args}) 
                      if img is not None else None
                      for img, args in zip(images, kwargs)]
        return images
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, pad_width={1}, pos={2}, mode={3})'.\
            format(self.size, self.pad_width, self.pos, self.mode)


def get_crop_width(input_size, output_size, pos='center'):
    output_size = output_size + input_size[len(output_size):]
    output_size = np.minimum(input_size, output_size)
    if pos == 'center':
        l = np.floor_divide(input_size - output_size, 2)
    elif pos == 'random':
        l = [np.random.randint(0, _ + 1) for _ in input_size - output_size]        
    return list(zip(l, input_size - output_size - l))


def crop(img, size=None, crop_width=None, pos='center', **kwargs):
    """ Crop the input numpy array image with crop_width and to given size.
    Args:
        img (numpy array): Image to be resized.
        size (tuple): Desired output size. 
        crop_width (list of tuples): Desired crop_width. 
        pos: one of {'center, 'random'}, default is 'center'. if given
             size, the parameter will decide whether to put original 
             image in the center or a random location.
        kwargs: other parameters in skimage.util.crop, use default just fine.
    
    crop_width and size can have same length as img, or 1d less than img.
    crop_width and size cannot be both None. If size = None, function will
    return image with img_size - crop_width. If crop_width = None, function 
    will return image with size. If both size and crop_width is not None,
    function will crop with crop_width first, then will try to meet size. 
    Function don't do any resize, rescale and pad process. Return img size 
    will be min(img.size-pad_width, size). 
    Returns:
        numpy array: Resized image.
    """
    copy = kwargs.setdefault('copy', False)
    order = kwargs.setdefault('order', 'K')
    
    if crop_width is not None:
        crop_width = crop_width + [(0, 0)] * (img.ndim - len(crop_width))
        img = skimage.util.crop(img, crop_width[:img.ndim], copy=copy, order=order)
    
    if size is not None:
        crop_var = get_crop_width(img.shape, output_size=size, pos=pos)
        img = skimage.util.crop(img, crop_var, copy=copy, order=order)
    return img


class Crop(object):
    def __init__(self, size=None, crop_width=None, pos='center', **kwargs):
        if size is not None:
            if isinstance(size, numbers.Number):
                size = (int(size), int(size))
            else:
                assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        
        self.size = size
        self.crop_width = crop_width
        self.pos = pos
        self.kwargs = kwargs
    
    def __call__(self, images, kwargs=None):
        if kwargs is None:
            kwargs = [{}] * len(images)
        if self.crop_width is not None:
            images = [crop(img, size=None, crop_width=self.crop_width, **{**self.kwargs, **args}) 
                      if img is not None else None
                      for img, args in zip(images, kwargs)]
        if self.size is not None:
            crop_width = get_crop_width(images[0].shape, output_size=self.size, pos=self.pos)
            images = [crop(img, size=None, crop_width=crop_width, **{**self.kwargs, **args}) 
                      if img is not None else None
                      for img, args in zip(images, kwargs)]
        return images
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_width={1}, pos={2})'.\
            format(self.size, self.crop_width, self.pos)


def hflip(img):
    return img[:, ::-1, ...]


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if np.random.random() < self.p:
            return [hflip(img) if img is not None else None for img in images]
        return images

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def vflip(img):
    return img[::-1, ...]


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if np.random.random() < self.p:
            return [vflip(img) if img is not None else None for img in images]
        return images

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def random_transform_pars(N, rotation=0., translate_x=0., translate_y=0., 
                          scale_x=0., scale_r=0., shear=0., 
                          projection_g=0., projection_h=0., p=0.5, seed=None):
    """ Randomly generate parameters for image transformation.
        If a scalar value is provided, the function will
        randomly generate N parameters inside the range
        If a list/array is provided, the function will use
        all combination of these values.
        
        # Returns
        A dictionary contains args for random transformation.
        Use get_transform_matrix to generate a transform matrix.
        And use transform to do affine/projective transformation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    pars = dict()
    
    # rotation
    rotation = (-rotation, rotation) if np.isscalar(rotation) else rotation
    r = np.random.uniform(rotation[0], rotation[1], N) * (np.random.random(N) < p)
    pars['rotation'] = r.tolist()
    
    # translation
    translate_x = (-translate_x, translate_x) if np.isscalar(translate_x) else translate_x
    tx = np.random.uniform(translate_x[0], translate_x[1], N) * (np.random.random(N) < p)
    translate_y = (-translate_y, translate_y) if np.isscalar(translate_y) else translate_y
    ty = np.random.uniform(translate_y[0], translate_y[1], N) * (np.random.random(N) < p)
    pars['translate'] = np.stack([tx, ty], axis=-1).tolist()
    
    # shear
    shear = (-shear, shear) if np.isscalar(shear) else shear
    s = np.random.uniform(shear[0], shear[1], N) * (np.random.random(N) < p)
    pars['shear'] = s.tolist()
    
    # scale
    scale_x = (1.*(1-scale_x), 1./(1-scale_x)) if np.isscalar(scale_x) else scale_x
    zx = np.random.uniform(np.log(scale_x[0]), np.log(scale_x[1]), N) * (np.random.random(N) < p)
    scale_r = (1.*(1-scale_r), 1./(1-scale_r)) if np.isscalar(scale_r) else scale_r
    zy = zx + (np.random.uniform(np.log(scale_r[0]), np.log(scale_r[1]), N) * (np.random.random(N) < p))
    pars['scale'] = np.stack([np.exp(zx), np.exp(zy)], axis=-1).tolist()
    
    # projection
    projection_g = (- projection_g, projection_g) if np.isscalar(projection_g) else projection_g
    pg = np.random.uniform(projection_g[0], projection_g[1], N) * (np.random.random(N) < p)
    projection_h = (- projection_h, projection_h) if np.isscalar(projection_h) else projection_h
    ph = np.random.uniform(projection_h[0], projection_h[1], N) * (np.random.random(N) < p)
    pars['projection'] = np.stack([pg, ph], axis=-1).tolist()
    
    return unpack_dict(pars, N)


def get_transform_matrix(rotation, translate, scale, shear, projection=(0, 0), center=(0, 0), inverse=True):
    """ Compute (inverse) matrix for affine/projective transformation
    .. Note::
        Affine transformation matrix is calculated as: M = T * C * RSS * C^-1
        T is translation matrix after rotation: [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
        C is translation matrix to keep center: [[1, 0, cx], [0, 1, cy], [0, 0, 1]]
        RSS is rotation with scale and shear matrix
        RSS(a, scale, shear) = [[cos(a)*scale_height, -sin(a + shear)*scale_height, 0],
                                [sin(a)*scale_width, cos(a + shear)*scale_width, 0],
                                [0, 0, 1]]
        The inverse matrix is M^-1 = C * RSS^-1 * C^-1 * T^-1
        Projective transformation: [[a, b, c], [d, e, f], [g, h, 1]], where g, h != 0
    Args:
        rotation (float or int): rotation angle in degrees between -180 and 180.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (list or tuple of floats): height_scale and width_scale
        shear (float): shear angle value in degrees between -180 to 180.
        center (tuple, optional): center offset in translation matrix
        projection (list or tuple of floats, optional): the projective transformation
        inverse (bool): apply inverse matrix (clockwise) or original matrix (anti-cloakwise)
    Returns:
        a 3*3 (inverse) matrix for affine/projective transformation
    """
    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"
    assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
        "Argument scale should be a list or tuple of length 2"
    assert isinstance(projection, (tuple, list)) and len(projection) == 2, \
        "Argument projection should be a list or tuple of length 2"
    assert isinstance(center, (tuple, list)) and len(center) == 2, \
        "Argument center should be a list or tuple of length 2"    
    
    rotation = math.radians(rotation)
    shear = math.radians(shear)
    
    if inverse:
        # Inverted rotation matrix with scale and shear
        d = math.cos(rotation + shear) * math.cos(rotation) + math.sin(rotation + shear) * math.sin(rotation)
        matrix = np.array([[math.cos(rotation + shear) / scale[1] / d, math.sin(rotation + shear) / scale[1] / d, 0],
                           [-math.sin(rotation) / scale[0] / d, math.cos(rotation) / scale[0] / d, 0], [0, 0, 1]])
    else:
        matrix = np.array([[math.cos(rotation + shear) * scale[0], -math.sin(rotation + shear) * scale[0], 0],
                           [math.sin(rotation) * scale[1],  math.cos(rotation) * scale[1], 0],
                           [0, 0, 1]])

    ## Offset center and apply translation: C * RSS^-1 * C^-1 * T^-1
    matrix[0, 2] = center[1] + matrix[0, 0] * (-center[1] - translate[1]) + matrix[0, 1] * (-center[0] - translate[0])
    matrix[1, 2] = center[0] + matrix[1, 0] * (-center[1] - translate[1]) + matrix[1, 1] * (-center[0] - translate[0])
    
    ## Add projection
    matrix[2, 0] = projection[0]
    matrix[2, 1] = projection[1]
        
    return matrix


def translate_offset_center(translate, input_size, output_size):  
    # offset matrix to the center of image
    center = (input_size[0] * 0.5 + 0.5, input_size[1] * 0.5 + 0.5)
    offset = (output_size[0] * 0.5 + 0.5, output_size[1] * 0.5 + 0.5)
    translate = (translate[0] + offset[0] - center[0], 
                 translate[1] + offset[1] - center[1])
    return center, offset, translate
    

def transform(img, matrix, size=None, out_dtype='image', **kwargs):
    """Apply affine/projective transformation on the image. 
    .. Note::
        image is centered under new size after affine transformation.
    Args:
        img (numpy array): input image.
        matrix (3*3 numpy array or a dictionary): provide either a transform matrix or pars to generate matrix.
        size (tuple, optional): the output image size.
        kwargs: parameters for get_transform_matrix and skimage.transform.warp. 
            get_transform_matrix args: [rotation, translate, scale, shear, projection, inverse]
            skimage.transform.warp functions:
            order: use order = 0 for mask to keep labels. This will avoid unnecessary post-treatment.
            mode and cval: fill area outside the transform with specific padding method/color.
            preserve_range: use preserve_range=True for higher order
    Return:
        images after transformation
    """
    out_dtype = img.dtype if out_dtype == 'image' else out_dtype
    if size is None:
        size = img.shape[:2]
    
    ## if no transform matrix is given, use default setting
    if matrix is None:
        matrix = {}
    if isinstance(matrix, dict):
        rotation = matrix.setdefault('rotation', 0.)
        translate = matrix.setdefault('translate', (0., 0.))
        scale = matrix.setdefault('scale', (0., 0.))
        shear = matrix.setdefault('shear', 0.)
        projection = matrix.setdefault('projection', (0., 0.))
        inverse = matrix.setdefault('inverse', True)
        
        # offset matrix to center
        center, _, translate = translate_offset_center(translate, input_size=img.shape[:2], output_size=size)
        # center = (img.shape[0] * 0.5 + 0.5, img.shape[1] * 0.5 + 0.5)
        # offset = (size[0] * 0.5 + 0.5, size[1] * 0.5 + 0.5)
        # translate = (translate[0] + offset[0] - center[0], translate[1] + offset[1] - center[1])
        matrix = get_transform_matrix(rotation, translate, scale, shear, 
                                      projection=projection, center=center, inverse=inverse)
    
    assert isinstance(matrix, np.ndarray) and matrix.shape == (3, 3), \
        "Invalid transform matrix"
    
    if not np.allclose(matrix, np.eye(3)):
        order = kwargs.setdefault('order', 1)
        mode = kwargs.setdefault('mode', 'constant')
        cval = kwargs.setdefault('cval', 0.0)
        clip = kwargs.setdefault('clip', True)
        preserve_range = kwargs.setdefault('preserve_range', False)
        
        if np.any(matrix[-1, :-1]):
            tform = skimage.transform.ProjectiveTransform(matrix=matrix)
        else:
            tform = skimage.transform.AffineTransform(matrix=matrix)
        img = skimage.transform.warp(img, tform, output_shape=size, order=order, 
                                     mode=mode, cval=cval, clip=clip, 
                                     preserve_range=preserve_range)
    return img_as(out_dtype)(img)


class RandomTransform(object):
    """ Random Transformation. 
    Argument:
        size: output image size
        rotation: float (degree)
        shear: float(degree)
        translate: tuple(x, y) 
        scale: tuple(zoom, h/w ratio)
        projection: tuple(x, y)
        inverse: use inverse transform or not
        p: probability for each transform
    """
    def __init__(self, size=None, rotation=0., translate=(0., 0.), 
                 scale=(0., 0.), shear=0., projection=(0., 0.), 
                 inverse=True, p=0.5, **kwargs):
        self.size = size
        self.rotation = rotation if rotation is not None else 0.
        self.shear = shear if shear is not None else 0.
        
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
        else:
            translate = (0., 0.)
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
        else:
            scale = (0., 0.)
        self.scale = scale
        
        if projection is not None:
            assert isinstance(projection, (tuple, list)) and len(projection) == 2, \
                "scale should be a list or tuple and it must be of length 2."
        else:
            projection = (0., 0.)
        self.projection = projection        

        self.inverse = inverse
        self.p = p
        self.kwargs = kwargs
    
    def get_params(self, input_size, output_size):
        pars = random_transform_pars(N=1, rotation=self.rotation, 
                                     translate_x=self.translate[0], translate_y=self.translate[1], 
                                     scale_x=self.scale[0], scale_r=self.scale[1], shear=self.shear, 
                                     projection_g=self.projection[0], projection_h=self.projection[1], 
                                     p=self.p, seed=None)[0]
        center, _, translate = translate_offset_center(pars['translate'], input_size, output_size)
        pars.update({'center': center, 'translate': translate, 'inverse': self.inverse})
        matrix = get_transform_matrix(**pars)
        return matrix, pars
    
    def __call__(self, images, kwargs=None):
        input_size = images[0].shape[:2]
        output_size = input_size if self.size is None else self.size
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        matrix, pars = self.get_params(input_size, output_size)
        
        if kwargs is None:
            kwargs = [{}] * len(images)
        
        def f(img, args={}):
            if img.ndim > 2:
                res = np.rollaxis(img, CHANNEL_AXIS)
                res = np.stack([transform(res[i], matrix, size=output_size, **{**self.kwargs, **args})
                                for i in range(len(res))], axis=CHANNEL_AXIS)
            else:
                res = transform(img, matrix, size=output_size, **{**self.kwargs, **args})
            return res
        return [f(img, args) if img is not None else None for img, args in zip(images, kwargs)]
        # return [transform(x, matrix, size=output_size, **self.kwargs) for x in images]
    
    def __repr__(self):
        s = '{name}(rotation={rotation}, translate={translate}, scale={scale}, shear={shear}, projection={projection})'
        return s.format(name=self.__class__.__name__, **dict(self.__dict__))


