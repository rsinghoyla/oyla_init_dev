import logging
import numpy as np

# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    
    High-frequency filters implemented:
        butterworth
        gaussian

    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H, D = None):
        H = np.fft.fftshift(H)
        #I_filtered = (self.a + self.b*H)*I
        LP = I-H*I
        if D is None:
            I_filtered = self.a*LP + (self.a+self.b)*H*I
        else:
            D = np.log1p(D)
            D = np.fft.fft2(D)
            I_filtered = self.a*LP/2.0+self.a*D/2.0 + (self.a+self.b)*H*I
        return I_filtered

    def get_illumination_reflectance(self,I,filter_params, filter_name='butterworth', H = None):
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        H = self._set_filter(filter_params,filter_name=filter_name, H = H, I_shape = I_fft.shape)
        H = np.fft.fftshift(H)
        
        reflectance = H*I_fft
        illumination = I_fft-reflectance
        reflectance = np.exp(np.real(np.fft.ifft2(reflectance)))
        illumination = np.exp(np.real(np.fft.ifft2(illumination)))
        return illumination, reflectance

    def _set_filter(self,filter_params, I_shape, filter_name='butterworth', H = None):
        """
        filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """
        
        # Filters
        if filter_name=='butterworth':
            H = self.__butterworth_filter(I_shape = I_shape, filter_params = filter_params)
        elif filter_name=='gaussian':
            H = self.__gaussian_filter(I_shape = I_shape, filter_params = filter_params)
        elif filter_name=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        return H
    
    def filter(self, I, filter_params, filter_name='butterworth', H = None,D=None):
        """
        Method to apply homormophic filter on an image

        Attributes:
            I: Single channel image
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        H = self._set_filter(filter_params,filter_name=filter_name, H = H, I_shape = I_fft.shape)
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H,D=D)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return I#np.uint8(I)
# End of class HomomorphicFilter

if __name__ == "__main__":
    import cv2
    import sys
    # Code parameters
    #path_in = '/home/images/in/'
    #path_out = '/home/images/out/'
    path_in = './'
    path_out = './'
    img_path = 'original.jpg'

    # Derived code parameters
    img_path_in = path_in + img_path
    if len(sys.argv) > 1:
        img_path_in = sys.argv[1]
    img_path_out = path_out + 'filtered.png'

    # Main code
    img = cv2.imread(img_path_in)[:, :, 0]
    homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
    img_filtered = homo_filter.filter(I=img, filter_params=[30,2])
    cv2.imwrite(img_path_out, img_filtered)
    
