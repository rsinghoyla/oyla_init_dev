import csv
import numpy as np
from matplotlib import cm
MODULATION_FREQUENCIES = np.asarray([24000, 12000, 6000, 3000, 1500, 750, 24000])*1000 # KHz to Hz
SPEED_LIGHT = 150000000.0
AMBIGUITY_DISTANCE_LUT = SPEED_LIGHT/MODULATION_FREQUENCIES*100  #in centi meters

def read_csv_parameters(csv_file):

    #####################################################################################
    # reads a csv file, row by row. Assumption is that its row transposed such that a row
    # lists all params and commands, and rows below list values taken by them.
    # Functions here will read the csv file, and set up a dictionary of commands from
    #           the rows of csv files.
    # There are four "Type" of commands/parameters,
    #           param: parameters for the supporting pythonic scripts
    #           adaptive_cmd: espros server command, takes "Value" from csv file, executed
    #                         at each epoch 
    #           logical_cmd: espros server command, takes value from param and code settings
    #           default_cmd: espros server command, takes value from csv file, executed
    #                        only once at begining of program
    #           na_cmd: ignored commands, GUI parameters
    #####################################################################################
    
    commands = {}

    with open(csv_file) as fp: 
        reader = csv.reader(fp) 
        for r in reader:
            if r[0] != '': 
                if r[0] in commands.keys(): 
                    if not isinstance(commands[r[0]][0],(list,)):  
                        commands[r[0]] = [commands[r[0]]] 
                    commands[r[0]].append(r[1:]) 
                else: 
                    commands[r[0]] = r[1:]

    # These are essential for further processing, i.e., 
    # Type, Key and Value should be in column 1 of sheet
    print(commands.keys())
    try:
        assert 'Type' in commands.keys()
        assert 'Key' in commands.keys()
        assert 'Value' in commands.keys()
    except:
        print("Check Column 1 of csv")
        return None
        # Only these types are supported
    try:
        assert np.unique(commands['Type']).tolist() == ['adaptive_cmd',  'default_cmd', 'filter_cmd',
                                                        'logical_cmd', 'na_cmd', 'param']
    except:
        print("Only certain cmds, and params are supported in Type")
        print(np.unique(commands['Type']).tolist())
        return None
    
    # convert multiple epoch values into an array
    commands['Value'] = np.asarray(commands['Value']).transpose().tolist()

    # grouping them together in groups of Type
    parameters = {}
    for i,c in enumerate(commands['Type']):
        if c not in parameters.keys():
            parameters[c] = {}
        parameters[c][commands['Key'][i]] = commands['Value'][i]
        #param['arg'] = 
        #parameters[c].append(param)
        
    return parameters

            
def convert_matrix_image(img, saturation_indices = None, no_data_indices = None, cmap = 'virdis', clim_min = None,
                         clim_max=None, outside_range_indices = None,colorize = False):

    ##########################################################################
    # converts a matrix of numbers into a 3D color image using colormap. Also assigns special color value to saturation
    # returns RGBA
    ##########################################################################

##  Use of the next two if loops is not clear - keeping in place in case of surprises
##    if clim_min:
##        img[np.where(img<clim_min)] = clim_min
##    if clim_max:
##        img[np.where(img>clim_max)] = clim_max
    
    img = img.astype('float32')
    
    img -= np.min(clim_min)
    img /= np.max(clim_max)
    img = np.uint8(cm.get_cmap(cmap)(img)*255)

    if no_data_indices:
        img[no_data_indices] = [0,0,0,255] # Color value to saturation
    if outside_range_indices:
        if colorize:
            img[outside_range_indices] = [135,0,175,255]#[0,0,0,255]
        else:
            img[outside_range_indices] = [0,0,0,255]
    if saturation_indices:
        if colorize:
            img[saturation_indices] = [255,255,255,255] # Color value to saturation
        else:
            img[saturation_indices] = [137,0,175,255] # Color value to saturation
    return img


def _transformation3_(inX, inY, inRCM, width, height, transform_types='cartesian', fov_angle=94, fov_angle_o = None):
                      
    
    '''
    To do spherical (i,j, rcm) into (x,y,z) where rcm is range in cm, i,j are pixel indices
    '''
    #width = IM.shape[0]
    #height = IM.shape[1]
    alfa0 = (fov_angle * 3.14159265359) / 360.0;
    step = 2*alfa0 / width
    
    if fov_angle_o is not None:
        beta0 = fov_angle_o * np.pi/ 360.0;  
        step_o = 2*beta0/height;
    else:
        step_o = step
        
    X=[]
    Y=[]
    Z=[]

    # dataPixelfield = IM
    for y,x,rcm in zip(inY,inX,inRCM):
        beta = (y-height/2) * step_o;
        #for x in X:
        alfa = (x-width/2) * step;
        #rcm =  (speedOfLight*ModFreq*1000)*(dataPixelfield[x,y]/30000.0)/10.0;

        if(transform_types =='cartesian'):
            #rcm = IM[x][y]
            X.append(rcm * np.cos(beta) * np.sin(alfa)+width/2)
            Y.append(rcm * np.sin(beta)+height/2)
            Z.append(rcm*np.cos(alfa)*np.cos(beta))
        else:
            X.append(x)
            Y.append(y)
            Z.append(rcm)

    x = np.array(X)
    y = np.array(Y)
    z = np.array(Z)
    return x, y, z

def some_common_utility(parameters,epoch_number):
    try:
        ambiguity_distance = AMBIGUITY_DISTANCE_LUT[int(parameters['adaptive_cmd']
                                                        ['setModulationFrequency'][epoch_number])]
    except ValueError:
        ambiguity_distance = None
    try:
        range_max = float(parameters['adaptive_cmd']['range_max'][epoch_number])*100
    except ValueError:
        range_max = None
    try:
        range_min = float(parameters['adaptive_cmd']['range_min'][epoch_number])*100
    except ValueError:
        range_min = None
    try:
        amplitude_min = float(parameters['adaptive_cmd']['setMinAmplitude'][epoch_number])
    except ValueError:
        amplitude_min = None
    try:
        saturation_flag = int(parameters['adaptive_cmd']['enableSaturation'][epoch_number])
    except ValueError:
        saturation_flag = None
    try:
        adc_flag = int(parameters['adaptive_cmd']['enableAdcOverflow'][epoch_number])
    except ValueError:
        adc_flag = None
    try:
        mod_freq = MODULATION_FREQUENCIES[int(parameters['adaptive_cmd']
                                              ['setModulationFrequency'][epoch_number])]
    except ValueError:
        mod_freq = None
    try:
       reflectivity_thresh =int(parameters['adaptive_cmd']['reflectivity_thresh'][epoch_number])
    except KeyError or ValueError:
        reflectivity_thresh = 0
    return ambiguity_distance, range_max, range_min, saturation_flag, adc_flag, mod_freq, amplitude_min,reflectivity_thresh


