##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.04.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version                                                             #
# Description                                                                                    #
#         utilties function like reading parameters from csv file, plotting,                     #
#                   animating frames to make movie                                               #
##################################################################################################


import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import matplotlib.animation as animation
#import matplotlib.pyplot as plt
import csv
import numpy as np

from matplotlib import cm

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
        assert np.unique(commands['Type']).tolist() == ['adaptive_cmd', 'default_cmd', 'logical_cmd', 'na_cmd', 'param']
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

def plotting_dist_ampl(dist_sin,ampl,parameters):
    #####################################################################################
    # Ignore for now
    #####################################################################################
    if 'cmap_distance' in parameters:
        cmap_distance = parameters['cmap_distance']
    else:
        cmap_distance = 'jet_r'

    if 'cmap_ampl' in parameters:
        cmap_ampl = parameters['cmap_ampl']
    else:
        cmap_ampl = 'gray'
    
    ax1 = pl.subplot(121)
    im1 = pl.imshow(dist_sin,aspect='equal',cmap=pl.get_cmap(cmap_distance))
    pl.axis('off')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    pl.colorbar(im1,cax=cax,extend='both')
    pl.clim(0,parameters['ambiguityDistance'])
    
    ax2 = pl.subplot(122)
    im2=pl.imshow(ampl,aspect='equal',cmap=pl.get_cmap(cmap_ampl))
    pl.axis('off')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    pl.colorbar(im2,cax=cax,extend='both')
    #pl.clim(0,ambiguityDistance)
    return im1, im2

def getComparedHDRAmplitudeImage_vectorized(amp,dist= None):
    highAmpThreshold = 2045
    Amp_case = np.where((amp[:,::2] > amp[:,1::2]), 1 , 2 )
     
    Amp_case = np.where(((amp[:,::2]>=highAmpThreshold )*(amp[:,1::2]<=highAmpThreshold)), 2 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<=highAmpThreshold )*(amp[:,1::2]>=highAmpThreshold)), 1 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<highAmpThreshold )*(amp[:,1::2]<highAmpThreshold)* (amp[:,::2] >= amp[:,1::2])), 1 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<highAmpThreshold )*(amp[:,1::2]<highAmpThreshold)* (amp[:,::2] < amp[:,1::2])), 2 , Amp_case )
    
    amp[:,::2] = np.where((Amp_case==2), amp[:,1::2] , amp[:,::2])
    amp[:,1::2] = np.where((Amp_case==1), amp[:,::2] , amp[:,1::2])
    
    if dist is not None:
        dist[:,::2] = np.where(((Amp_case==2)*(dist is not None)), dist[:,1::2] , dist[:,::2])
        dist[:,1::2] = np.where(((Amp_case==1)*(dist is not None)), dist[:,::2] , dist[:,1::2])
    return amp, dist

def getComparedHDRAmplitudeImage(amp,dist= None):
    highAmpThreshold = 2045
    Amp_case = np.where((amp[:,::2] > amp[:,1::2]), 1 , 2 )
     
    Amp_case = np.where(((amp[:,::2]>=highAmpThreshold )*(amp[:,1::2]<=highAmpThreshold)), 1 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<=highAmpThreshold )*(amp[:,1::2]>=highAmpThreshold)), 2 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<highAmpThreshold )*(amp[:,1::2]<highAmpThreshold)* (amp[:,::2] >= amp[:,1::2])), 1 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<highAmpThreshold )*(amp[:,1::2]<highAmpThreshold)* (amp[:,::2] < amp[:,1::2])), 2 , Amp_case )
    
    amp[:,::2] = np.where((Amp_case==2), amp[:,1::2] , amp[:,::2])
    amp[:,1::2] = np.where((Amp_case==1), amp[:,::2] , amp[:,1::2])
    
    dist[:,::2] = np.where(((Amp_case==2)*(dist is not None)), dist[:,1::2] , dist[:,::2])
    dist[:,1::2] = np.where(((Amp_case==1)*(dist is not None)), dist[:,::2] , dist[:,1::2])
    return amp, dist
# def getComparedHDRAmplitudeImage(amp, dist=None):
#     ##########################################################################
#     # This is copied from pixelCorrector.cpp in espros 
#     # It changes the readout distance using amplitude data
#     ##########################################################################
    
#     highAmpThreshold = 2045;

#     for i in range(0,amp.shape[0]): 
#         for j in range(0,amp.shape[1],2):
#             amp1 = amp[i][j];
#             amp2 = amp[i][j+1];
#             ampCase = 1;
             
#             if (amp1 < highAmpThreshold and amp2 < highAmpThreshold): # both amplitudes in range
#                 if amp2 > amp1: ampCase = 2;
#                 else:           ampCase = 1;
#             elif (amp1 >= highAmpThreshold and amp2 <= highAmpThreshold):  #one amplitude in range or both amplitudes LOW_AMPLITUDE
#                 ampCase = 2;
#             elif (amp1 <= highAmpThreshold and amp2 >= highAmpThreshold): #one amplitude in range or both amplitudes LOW_AMPLITUDE
#                 ampCase = 1;
#             else: #both amplitudes out of range
#                 if(amp2 < amp1): ampCase = 1;
#                 else:            ampCase = 2;
                

#             if(ampCase == 1):
#                 if dist is not None:
#                     dist[i][j+1] = dist[i][j];    
#                 amp[i][j+1] = amp[i][j];
#             else:
#                 if dist is not None:
#                     dist[i][j] = dist[i][j+1];
#                 amp[i][j] = amp[i][j+1];
#     return amp, dist

def plotting(to_display, imaging_type, ambiguity_distance = None, range_max = 9 ,
             range_min = 0.1, amplitude_min = None, saturation_flag = None ,adc_flag = None,saturation = 65400,
             canvas_figure = None):
    
    assert len(to_display) == len(imaging_type)
    #####################################################################################
    # plotting a list of images depending on their imaging type
    # inputs list of images in to_display and corresponding list of imaging types in imaging_type
    #####################################################################################

    #by default each imaging_type in X+y will be plotted in a column
    number_columns = len(imaging_type)
    number_rows = 1

    #for each imaging_type depending on number of frames we will adjust the rows,columns of subplots
    if 'Dist_Ampl' in imaging_type:
        number_columns += 1
    if 'DCS' in imaging_type:
        number_rows += 1
        number_columns += 1
    if 'DCS_2' in imaging_type:
        number_rows += 1
    if 'HDR' in imaging_type:
        number_rows += 1
        number_columns += 1

    # for each imaging_type display in subplot; here is where logic for scaling/thresholding will come in
    for ind in range(len(imaging_type)):
        
        if imaging_type[ind] == 'Gray':

            grayscale_saturated_indices = np.where(np.squeeze(to_display[ind]>=4095))
            print("Grayscale Saturated indices =", np.count_nonzero(grayscale_saturated_indices))
            img = np.squeeze(to_display[ind])
            plot_subplot(number_rows,number_columns,ind+1, img,cmap='gray',title='Grayscale', enable_colorbar=True,
                         clim_min=2048, clim_max=4095, saturation_indices = grayscale_saturated_indices,
                         canvas_figure = canvas_figure)

        elif imaging_type[ind] == 'Dist_Ampl':
            
            #####to scale and threshold#######################################################################
            #print("sum,max before ", np.sum(to_display[ind]))

            amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,1])>=65400)
            print("Amplitude Saturated indices =", np.count_nonzero(amplitude_saturated_indices))

            if saturation_flag:
                to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]==65400)] = 0
            if adc_flag:
                to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]==65500)] = 0
                        
            #print("sum,max after saturation and adc ", saturation_flag, adc_flag, np.sum(to_display[ind]),np.max(to_display[ind]))
            #print(amplitude_min)

##            if amplitude_min:
##                to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]<=amplitude_min)] = 0
##                to_display[ind][:,:,1][np.where(to_display[ind][:,:,1]<=amplitude_min)] = 0

            low_amplitude_indices = np.where(np.squeeze(to_display[ind][:,:,1])==65300)
            print("Low Amplitude Indices =", np.count_nonzero(low_amplitude_indices))

##            to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]==65300)] = 0
##            to_display[ind][:,:,1][np.where(to_display[ind][:,:,1]==65300)] = 0

            #print("sum,max after LSB ", amplitude_min, np.sum(to_display[ind]))

            img = np.squeeze(to_display[ind])[:,:,0]

            if ambiguity_distance:
                img = img/(30000.0)*ambiguity_distance
                #print("sum,max after scaling to ambiguity distance",ambiguity_distance,np.sum(img),np.max(img))
                if range_max:
                    img[np.where(img>range_max)] = 0
                if range_min:
                    img[np.where(img<range_min)] = 0
                #print("sum,max after range ",range_max, range_min ,np.sum(img),np.max(img))
                    
            ####################################################################################################
                 
            plot_subplot(number_rows, number_columns, ind+1, img,cmap='jet_r', title = 'Distance',enable_colorbar = True,
                         clim_min=range_min, clim_max=range_max,saturation_indices = amplitude_saturated_indices,
                         no_data_indices = low_amplitude_indices,canvas_figure = canvas_figure)
            img = np.squeeze(to_display[ind])[:,:,1]
            plot_subplot(number_rows, number_columns, ind+1+number_rows, img,cmap='jet', title='Amplitude',
                         enable_colorbar = True, clim_min=1, clim_max=2000, saturation_indices = amplitude_saturated_indices,
                         no_data_indices = low_amplitude_indices, canvas_figure= canvas_figure)

        elif imaging_type[ind] == 'Dist':

            img = np.squeeze(to_display[ind])
            if ambiguity_distance:
                img = img/(30000.0)*ambiguity_distance
            distance_saturated_indices = np.where(np.squeeze(to_display[ind])==saturation)
            img = np.squeeze(to_display[ind])

            plot_subplot(number_rows,number_columns,ind+1, img, cmap='viridis',title = 'Distance', enable_colorbar = True,
                         clim_min=0.5, clim_max=5,saturation_indices = distance_saturated_indices,
                         canvas_figure = canvas_figure)

        elif imaging_type[ind] == 'Ampl':
            to_display[ind][np.where(to_display[ind]==65300)] = 0
            img = np.squeeze(to_display[ind])
            distance_saturated_indices = np.where(np.squeeze(to_display[ind])==saturation)

            plot_subplot(number_rows,number_columns,ind+1, img,cmap='gnuplot_r',title = 'Amplitude', enable_colorbar = True,
                         clim_min=1, clim_max=2000, saturation_indices = distance_saturated_indices,
                         canvas_figure = canvas_figure)

        elif imaging_type[ind] == 'DCS' :
            
            img = np.squeeze(to_display[ind])[:,:,0]
            median_scale =  np.median(img)
            plot_subplot(number_rows, number_columns, ind+1, img,title = '0 Deg',cmap='viridis',enable_colorbar = True,
                         clim_min=median_scale-500, clim_max=median_scale+500,canvas_figure = canvas_figure)
            img = np.squeeze(to_display[ind])[:,:,1]
            plot_subplot(number_rows, number_columns, ind+2, img, title = '90 Deg', cmap='viridis',enable_colorbar = True,
                           clim_min=median_scale-500, clim_max=median_scale+500,canvas_figure = canvas_figure)
            img = np.squeeze(to_display[ind])[:,:,2]
            plot_subplot(number_rows, number_columns, ind+1+number_columns, img,title ='180 Deg', cmap='viridis',
                         enable_colorbar = True, clim_min=median_scale-500, clim_max=median_scale+500,
                         canvas_figure = canvas_figure)
            img = np.squeeze(to_display[ind])[:,:,0] + np.squeeze(to_display[ind])[:,:,2]

            plot_subplot(number_rows, number_columns, ind+2+number_columns, img, title = '0+180 Deg', cmap='viridis',
                         enable_colorbar = True, clim_min=((2*median_scale)-200), clim_max=((2*median_scale)+200),
                         canvas_figure = canvas_figure)

        elif imaging_type[ind] == 'HDR':
            
            img = np.squeeze(to_display[ind])[:,:,0]
            amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,0])>=saturation)
            ## if manipulating img directly (as is done on the basis of range min/max below) saturation indices must be identified before zero distance indices
            if ambiguity_distance:
                img = img/(30000.0)*ambiguity_distance
                if range_max:
                    img[np.where(img>range_max)] = 0
                if range_min:
                    img[np.where(img<range_min)] = 0
            zero_distance_indices = np.where(np.squeeze(img)==0)
            plot_subplot(number_rows, number_columns, ind+1, img, cmap='jet_r',title = 'High Exposure Time',
                         enable_colorbar = True, clim_min=range_min, clim_max=range_max,
                         saturation_indices = amplitude_saturated_indices, no_data_indices = zero_distance_indices,
                         canvas_figure = canvas_figure)

            img = np.squeeze(to_display[ind])[:,:,1]
            amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,1])>=saturation)
            if ambiguity_distance:
                img = img/(30000.0)*ambiguity_distance
                if range_max:
                    img[np.where(img>range_max)] = 0
                if range_min:
                    img[np.where(img<range_min)] = 0
            zero_distance_indices = np.where(np.squeeze(img)==0)
            plot_subplot(number_rows, number_columns, ind+2, img, cmap='jet_r', title = 'Low Exposure Time',
                         enable_colorbar = True, clim_min=range_min, clim_max=range_max,
                         saturation_indices = amplitude_saturated_indices, no_data_indices = zero_distance_indices,
                         canvas_figure = canvas_figure)

            # img = np.squeeze(to_display[ind])[:,:,2]
            # amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,2])>=saturation)
            # if ambiguity_distance:
            #     img = img/(30000.0)*ambiguity_distance
            #     if range_max:
            #         img[np.where(img>range_max)] = 0
            #     if range_min:
            #         img[np.where(img<range_min)] = 0
            # zero_distance_indices = np.where(np.squeeze(img)==0)   
            # plot_subplot(number_rows, number_columns, ind+1+number_columns, img, cmap='jet_r', title ='Interleaved Image',
            #              enable_colorbar = True, clim_min=range_min, clim_max=range_max,
            #              saturation_indices = amplitude_saturated_indices,no_data_indices = zero_distance_indices,
            #              canvas_figure = canvas_figure)

            # img = np.squeeze(to_display[ind])[:,:,3]
            # ampl = np.squeeze(to_display[ind])[:,:,4]
            # #ampl,img = getComparedHDRAmplitudeImage(ampl,img)
            # amplitude_saturated_indices = np.where(np.squeeze(ampl>=saturation))
            
            # if ambiguity_distance:
            #     img = img/(30000.0)*ambiguity_distance
            #     if range_max:
            #         img[np.where(img>range_max)] = 0
            #     if range_min:
            #         img[np.where(img<range_min)] = 0
            # zero_distance_indices = np.where(np.squeeze(img)==0)     
            # plot_subplot(number_rows, number_columns, ind+2+number_columns, img, cmap='jet_r', title ='Combined Image',
            #              enable_colorbar = True, clim_min=range_min, clim_max=range_max,
            #              saturation_indices = amplitude_saturated_indices, no_data_indices = zero_distance_indices,
            #              canvas_figure = canvas_figure)

            # tmp = np.zeros((2*img.shape[0]+100,img.shape[1]))
            # tmp[:img.shape[0],:] = img
            # img = np.squeeze(to_display[ind])[:,:,2]
            # tmp[img.shape[0]+100:,:] = img
            # plot_subplot(1,1,1,tmp)

        elif imaging_type[ind] == 'DCS_2':

            img = np.squeeze(to_display[ind])[:,:,0]
            plot_subplot(number_rows, number_columns, ind+1, img)
            img = np.squeeze(to_display[ind])[:,:,1]
            plot_subplot(number_rows, number_columns, ind+1+number_columns, img)
            
def convert_matrix_image(img, saturation_indices = None, no_data_indices = None, cmap = 'virdis', clim_min = None,
                         clim_max=None):

    ##########################################################################
    # converts a matrix of numbers into a 3D color image using colormap. Also assigns special color value to saturation
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
    if saturation_indices:
        img[saturation_indices] = [135,0,175,255] # Color value to saturation
    return img

def plot_subplot(number_rows, number_columns, ind, img, cmap='viridis',title='',enable_colorbar=True,
                 clim_min=None, clim_max=None,saturation_indices = None, no_data_indices = None,canvas_figure = None):

    ##########################################################################
    # simple function that adds a subplot depending on rows, columns and index
    ##########################################################################


    img = np.flipud(convert_matrix_image(img,cmap= cmap, clim_min=clim_min, clim_max=clim_max,
                                         saturation_indices = saturation_indices,no_data_indices = no_data_indices))
    
    ax  = canvas_figure.add_subplot(number_rows,number_columns,ind)
    ax.set_title(title)
    if img.ndim == 2:
        im = ax.imshow(img.transpose(),aspect='equal',cmap=cmap)
    elif img.ndim == 3:
        im = ax.imshow(img.transpose(1,0,2),aspect='equal', cmap=cmap)
    ax.axis('off')
    #ax.subplots_adjust(wspace=0.15,  hspace=0.15)
    #ax.subplots_adjust(bottom=0.1, right=0.95, top=0.9, left = 0.01)
    if enable_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.5%', pad=0.1)
        im.figure.colorbar(im,cax=cax,extend='both')
    #im.figure.colorbar(im,cax=cax,extend_both)
    if clim_min and clim_max:
        im.set_clim(clim_min,clim_max)
    #pl.subplots_adjust(wspace = 0.0)
    ax.figure.canvas.draw()
#from pylab import *



