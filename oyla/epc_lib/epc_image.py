#-------------------------------------------------------------------------------
# Name:        epc_image
# Purpose:     Python class for project-independent stuff related to image
#              acquisition
# Version:     3.0
# Created:     28.01.2019
# Authors:     sho
# Copyright:   (c) ESPROS Photonics AG, 2016

# Oyla modifications
# getDistAmplHDR for getting the HDR data based on code in the client
# each getXXX input is the frame number and its output is [data, serverIP, data_type and frame_number]
#-------------------------------------------------------------------------------


# Official Libraries
import sys
import numpy as np
import socket
import struct
import time
class epc_image:

    def __init__(self, epc_server):
        self._server = epc_server

    def getDCSs(self,frame_number=None,is_gray=False):
        s = socket.create_connection((self._server.IP, self._server.port))
        command = 'getDCSSorted\n'
        s.send(command.encode())

        imageDataVector = bytearray()
        remaining = self._imageSizeBytesAllDCSs
        while remaining > 0:
            chunk = s.recv(remaining)       # Get available data.
            imageDataVector.extend(chunk)   # Add to already stored data.
            remaining -= len(chunk)

        s.close()
        _im = self._imageVectorToArray(imageDataVector, self._numberOfImageDataFrame)
        if is_gray:
            return [_im, self._server.IP, frame_number,'Gray']
        else:
            if self._numberOfImageDataFrame==4:
                return [_im, self._server.IP, frame_number,'DCS']
            else:
                return [_im, self._server.IP, frame_number,'DCS_2']
        
    def getGray(self,frame_number=None):
        return self.getDCSs(frame_number,is_gray=True)
    
    def getDistAmpl(self,frame_number=None,mti=None,xSample = None, ySample = None, mSample = None):
        s = socket.create_connection((self._server.IP, self._server.port))
        command='getDistanceAndAmplitudeSorted\n'
        s.send(command.encode())
        if mti is not None:
            print("Sending pulse.",len(xSample))#s,mti.GetDeviceParam(MTIParam.SampleRate))
            mti.StopDataStream()
            mti.SendDataStream(xSample,ySample,mSample,len(xSample),0,False)
            mti.StartDataStream(1, False)
        imageDataVector = bytearray()
        remaining = (self._imageSizeBytes) * 2
        first = True
        while remaining > 0:
            chunk = s.recv(remaining)       # Get available data.
            imageDataVector.extend(chunk)   # Add to already stored data.
            remaining -= len(chunk)
            if first and len(chunk)>0:
                first = False
                #print("first response from server",time.time())
        s.close()
        _im = self._imageVectorToArray(imageDataVector, 2)
        return [_im, self._server.IP, frame_number,'Dist_Ampl']
    
    def getDistAmplHDR(self,frame_number=None):
        s = socket.create_connection((self._server.IP, self._server.port))
        command='getDistanceAndAmplitudeSorted\n'
        s.send(command.encode())

        imageDataVector = bytearray()
        remaining = (self._imageSizeBytes) * 2
        while remaining > 0:
            chunk = s.recv(remaining)       # Get available data.
            imageDataVector.extend(chunk)   # Add to already stored data.
            remaining -= len(chunk)

        s.close()
        data = imageDataVector
        dataInt0Raw = bytearray()
        dataInt1Raw = bytearray()
        dataInt2Raw = bytearray()
        dataInt3Raw = bytearray()
        rows = 240 #hard coded
        width = 320 #hard coded
        ampl = data[rows*width*2:]
        data = data[:rows*width*2]
        for i in range(rows//2):
            if i%2:
                #print(i)
                dataInt0Raw.extend(data[i*width*2:(i+1)*width*2])
                dataInt0Raw.extend(data[i*width*2:(i+1)*width*2])
                #print(len(dataInt0Raw))
            else:
                dataInt1Raw.extend(data[i*width*2:(i+1)*width*2])
                dataInt1Raw.extend(data[i*width*2:(i+1)*width*2])
                #print(len(dataInt1Raw))
        for i in range(rows//2,rows):
            if i%2:
                dataInt1Raw.extend(data[i*width*2:(i+1)*width*2])
                dataInt1Raw.extend(data[i*width*2:(i+1)*width*2])
            else:
                dataInt0Raw.extend(data[i*width*2:(i+1)*width*2])
                dataInt0Raw.extend(data[i*width*2:(i+1)*width*2])
        for i in range(rows):
            dataInt2Raw.extend(data[i*width*2:(i+1)*width*2])
            dataInt3Raw.extend(data[i*width*2:(i+1)*width*2])

        I = np.zeros((width,rows,5),dtype='uint16')
        I[:,:,0] = np.squeeze(self._imageVectorToArray(dataInt0Raw, 1))
        I[:,:,1] = np.squeeze(self._imageVectorToArray(dataInt1Raw, 1))
        I[:,:,2] = np.squeeze(self._imageVectorToArray(dataInt2Raw, 1))
        I[:,:,3] = np.squeeze(self._imageVectorToArray(dataInt3Raw, 1))
        I[:,:,4] = np.squeeze(self._imageVectorToArray(ampl, 1))
        #print(np.mean(I,axis=(0,1)))
        return [I, self._server.IP, frame_number,'HDR']
    

    def getDist(self,frame_number=None):
        s = socket.create_connection((self._server.IP, self._server.port))
        command='getDistanceSorted\n'
        s.send(command.encode())

        imageDataVector = bytearray()
        remaining = (self._imageSizeBytes) * 1
        while remaining > 0:
            chunk = s.recv(remaining)       # Get available data.
            imageDataVector.extend(chunk)   # Add to already stored data.
            remaining -= len(chunk)

        s.close()
        _im = self._imageVectorToArray(imageDataVector, 1)
        return [_im, self._server.IP, frame_number,'Dist']
    
    def getAmpl(self,frame_number=None):
        s = socket.create_connection((self._server.IP, self._server.port))
        command='getAmplitudeSorted\n'
        s.send(command.encode())

        imageDataVector = bytearray()
        remaining = (self._imageSizeBytes) * 1
        while remaining > 0:
            chunk = s.recv(remaining)       # Get available data.
            imageDataVector.extend(chunk)   # Add to already stored data.
            remaining -= len(chunk)

        s.close()
        _im = self._imageVectorToArray(imageDataVector, 1)
        return [_im, self._server.IP, frame_number,'Ampl']

    def getTemperature(self,frame_number=None):
        s = socket.create_connection((self._server.IP, self._server.port))
        command='getTemperature\n'
        s.send(command.encode())
        tempVector = bytearray()
        chunk = s.recv(2)       # Get available data.
        tempVector=chunk
        s.close()

        unpackedString = 'H' * (int(tempVector.__len__()/2)) # signed short (16bit)
        tempData16bit = list(struct.unpack('<'+unpackedString, tempVector)) # little endian

        return [tempData16bit,self._server.IP,frame_number,'Temp']


    def setNumberOfRecordedColumns(self, NbrMeasCols):
        self._numberOfColumns  = NbrMeasCols

    def getNumberOfRecordedColumns(self):
        return self._numberOfColumns

    def setNumberOfRecordedRows(self, NbrMeasRows):
        self._numberOfRows  = NbrMeasRows

    def getNumberOfRecordedRows(self):
        return self._numberOfRows

    def setNumberOfRecordedImageDataFrames(self, NbrMeasDataFrame):
        self._numberOfImageDataFrame = NbrMeasDataFrame

    def getNumberOfRecordedImageDataFrames(self):
        return self._numberOfImageDataFrame

    def updateNbrRecordedBytes(self):
        self._imageSizeBytes        = self._numberOfRows * self._numberOfColumns * 2
        self._imageSizeBytesAllDCSs = self._numberOfImageDataFrame * self._imageSizeBytes

    def _imageVectorToArray(self, imageDataVector, numberOfElements):
        imageDataVector = bytes(imageDataVector)

        unpackedString = 'H' * (int(imageDataVector.__len__() / 2))  # signed short (16bit)

        imageData16bit = list(struct.unpack('<' + unpackedString, imageDataVector))  # little endian

        # Store data directly as numpy array:
        imageData = np.transpose(np.reshape(np.array(imageData16bit, dtype='uint16'), (numberOfElements, self._numberOfRows, self._numberOfColumns)), [2, 1, 0])

        return imageData
