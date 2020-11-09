#from oyla.mvc.utils import CAMERA_VERSION, FOV
from oyla.utils import read_csv_parameters, some_common_utility
import json
import numpy as np

def read_input_config(args):
    
    input_data_folder_name =  args.input_dir#"/".join(args.input_dir.split("/")[:-1])
    csv_file = input_data_folder_name+'/parameters.csv'
    parameters = read_csv_parameters(csv_file)

    _dim = list(map(int,parameters['adaptive_cmd']['setROI'][0].split(' ')))
    width = _dim[1]-_dim[0]+1
    height = _dim[3]-_dim[2]+1
    if parameters['param']['enableVerticalBinning'][0] == '1':
        height = height // 2
    if parameters['param']['enableHorizontalBinning'][0] == '1':
        width = width //2
    if len(parameters['param']['chip_id'][0].split(',')) == 2:
        width = width *2
    print("height, width",height, width)

    imaging_type = parameters['param']['imaging_type'][0].split('+')
    imaging_mode = parameters['param']['imaging_mode'][0]
    imaging_type = imaging_type[0]
    assert imaging_type == 'Dist_Ampl', "Supports only dist ampl data"

    with open(input_data_folder_name+'/commandline_args.txt','r') as fp:
        commandline_args = json.load(fp)
    print(commandline_args)

    if 'camera_version' in parameters['param']:
            camera_version = parameters['param']['camera_version'][0]
    if 'camera_version' in commandline_args:        
        if commandline_args['camera_version']  is not None:
            camera_version = commandline_args['camera_version'] 


    #assert camera_version in CAMERA_VERSION, "Not supported camera version"
    fov_x = commandline_args['fov_x'] 
    fov_y = commandline_args['fov_y'] 

    # consistent with controller
    ambiguity_distance, range_max, range_min, saturation_flag, adc_flag, mod_freq, ampl_min,reflectivity_thresh = some_common_utility(parameters,0)
    z_max = range_max
    z_min = range_min
    if args.z_min is not None:
            z_min = args.z_min
    if args.z_max is not None:
            z_max = args.z_max

    y_max = args.y_max
    y_min = args.y_min
    x_max = args.x_max
    x_min = args.x_min
    
    if args.y_max is None:
        y_max = z_max*np.sin(fov_y/360*np.pi)+height/2
    if args.y_min is None:
        y_min = -y_max

    if args.x_max is None:
        x_max = z_max*np.sin(fov_x/360*np.pi)+width/2
    if args.x_min is None:
        x_min = -x_max
    return x_max, x_min, y_max, y_min, z_max, z_min, range_max, range_min
