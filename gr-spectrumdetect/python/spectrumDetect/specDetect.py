#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-spectrumDetect author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
import torchsig
import json
from gnuradio import gr
import ultralytics
from ultralytics import YOLO
import torch
import torchaudio
import pmt
from .rxTime import rxTime
import time
import sigmf
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
from datetime import datetime
from math import pi, e
from torchsig.models import XCiTClassifier
import cv2

class specDetect(gr.sync_block):

    def __init__(self, centerFrequency, sampleRate, vectorSize, nfft, trainedWidebandModel, trainedNarrowbandModel, augment, iou, conf, agnosticNms, maxDet, writeLabeledWBImages, writeWBImages, writeWBIQFile, writeNBIQFile, gpuDevice, gpuHalf, detectJson, wbDetectOnly):
        gr.sync_block.__init__(self,
            name="specDetect",
            in_sig=[(np.complex64,vectorSize)],
            out_sig=None)
        self.detectJson = detectJson    
        self.fc = centerFrequency    
        self.fs = sampleRate            
        self.nfft = nfft
        self.nfftSamples = vectorSize
        self.gpuDevice = gpuDevice
        self.gpuHalf = gpuHalf
        self.wb_model_path = trainedWidebandModel    
        self.wb_model = YOLO(self.wb_model_path)
        self.wb_detect_only = wbDetectOnly
        if self.gpuDevice == 'cpu':
            self.wb_model.to('cpu')
            self.gpuHalf = False
        elif self.gpuDevice:
            torch.set_default_device('cuda'+':'+str(int(self.gpuDevice)))
            self.wb_model.to('cuda'+':'+str(int(self.gpuDevice)))         
        self.portName1 = "detect_pmt"
        self.message_port_register_out(pmt.intern(self.portName1))
        self.d_rxTime = rxTime(vlen=nfft)
        self.augment = augment
        self.iou = iou
        self.conf = conf
        self.agnosticNms = agnosticNms
        self.maxDet = maxDet
        self.use_PPS_time = False
        self.upper_limit = np.floor(self.fc + self.fs/2.0)
        self.lower_limit = np.floor(self.fc - self.fs/2.0)
        self.writeLabeledWBImages = writeLabeledWBImages
        self.writeWBImages = writeWBImages
        self.writeWBIQFile = writeWBIQFile
        self.writeNBIQFile = writeNBIQFile
        self.trainedNarrowbandModel = trainedNarrowbandModel
        if self.trainedNarrowbandModel == None or self.trainedNarrowbandModel == "None" or self.trainedNarrowbandModel == "":
            self.trainedNarrowbandModel = None
        if self.trainedNarrowbandModel != None:
            self.nb_model_mod_rec = XCiTClassifier.load_from_checkpoint(checkpoint_path=self.trainedNarrowbandModel)
            if self.gpuDevice == 'cpu':
                self.nb_model_mod_rec.to('cpu')
                self.gpuHalf = False
            elif self.gpuDevice:
                self.nb_model_mod_rec.to('cuda'+':'+str(int(self.gpuDevice)))  
                if self.gpuHalf == True:
                    self.nb_model_mod_rec.half() 

        self.nb_mod_rec_list: list = [
            "ook",
            "bpsk",
            "4pam",
            "4ask",
            "qpsk",
            "8pam",
            "8ask",
            "8psk",
            "16qam",
            "16pam",
            "16ask",
            "16psk",
            "32qam",
            "32qam_cross",
            "32pam",
            "32ask",
            "32psk",
            "64qam",
            "64pam",
            "64ask",
            "64psk",
            "128qam_cross",
            "256qam",
            "512qam_cross",
            "1024qam",
            "2fsk",
            "2gfsk",
            "2msk",
            "2gmsk",
            "4fsk",
            "4gfsk",
            "4msk",
            "4gmsk",
            "8fsk",
            "8gfsk",
            "8msk",
            "8gmsk",
            "16fsk",
            "16gfsk",
            "16msk",
            "16gmsk",
            "ofdm-64",
            "ofdm-72",
            "ofdm-128",
            "ofdm-180",
            "ofdm-256",
            "ofdm-300",
            "ofdm-512",
            "ofdm-600",
            "ofdm-900",
            "ofdm-1024",
            "ofdm-1200",
            "ofdm-2048",
            "fm",
            "am-dsb-sc",
            "am-dsb",
            "am-lsb",
            "am-usb",
            "lfm_data",
            "lfm_radar",
            "chirpss",
        ]


    def baseband_downsample_complex_to_complex(self,input_data,r_cfreq,signal_freq,israte,osrate,truncate):
        if self.gpuDevice != 'cpu':
            with torch.cuda.device('cuda'+':'+str(int(self.gpuDevice))):
                input_data.to('cuda'+':'+str(int(self.gpuDevice)))
                x = torch.linspace(0,len(input_data)-1,len(input_data),dtype=torch.complex64,device='cuda'+':'+str(int(self.gpuDevice)))
                fshift = (r_cfreq-signal_freq)/(israte*1.0)
                fv = e**(1j*2*pi*fshift*x)
                fv.to('cuda'+':'+str(int(self.gpuDevice)))
                input_data = input_data * fv
                num_samples_in = len(input_data)
                num_samples = int(np.ceil(num_samples_in/(israte/osrate)))
                osrate_new = int((num_samples*israte)/num_samples_in)
                down_factor = int(israte/osrate_new)
                transform = torchaudio.transforms.Resample(int(israte/osrate_new),1,dtype=torch.float32).to('cuda'+':'+str(int(self.gpuDevice)))
                if truncate:
                    test = transform(input_data.real) + 1j*transform(input_data.imag)
                    return test[:4096],israte/down_factor, True
                else:
                    test = transform(input_data.real) + 1j*transform(input_data.imag)
                    return test,israte/down_factor, True
        elif self.gpuDevice == 'cpu':
            with torch.cuda.device('cpu'):
                input_data.to('cpu')
                x = torch.linspace(0,len(input_data)-1,len(input_data),dtype=torch.complex64,device='cpu')
                fshift = (r_cfreq-signal_freq)/(israte*1.0)
                fv = e**(1j*2*pi*fshift*x)
                fv.to('cpu')
                input_data = input_data * fv
                num_samples_in = len(input_data)
                num_samples = int(np.ceil(num_samples_in/(israte/osrate)))
                osrate_new = int((num_samples*israte)/num_samples_in)
                down_factor = int(israte/osrate_new)
                transform = torchaudio.transforms.Resample(int(israte/osrate_new),1,dtype=torch.float32).to('cpu')
                if truncate:
                    test = transform(input_data.real) + 1j*transform(input_data.imag)
                    return test[:4096],israte/down_factor, False
                else:
                    test = transform(input_data.real) + 1j*transform(input_data.imag)
                    return test,israte/down_factor, False                    

                
    def work(self, input_items, output_items):


        num_input_items = len(input_items[0])        
        in0 = input_items[0]
        nread = self.nitems_read(0)
        tags = self.get_tags_in_range(0, nread, nread + num_input_items)

        if (self.use_PPS_time == False):
            # PPS time has not been initialized yet, so search to find an rx_time tag. the idea
            # is to lock into PPS time if it becomes available
            for tag in tags:
                keyString = pmt.to_python(tag.key)
                if (keyString == 'rx_time'):
                    # now using PPS_time, and never have to rerun the search to see if rx_time
                    # tag is avaialble, now switched over to always expecting 1 PPS
                    self.use_PPS_time = True
                    # exit from the search
                    break

        # now using PPS time, so current_time based on it
        if (self.use_PPS_time):
            self.d_rxTime.processTags(tags)
            current_time = self.d_rxTime.getNanoSecondsSinceEPOC(nread)
        else:
            # PPS time not yet initialized, so use current system time
            current_time = np.uint64(time.time()*1e9)

        # will run torchsig app when:
        #    1. PPS time is being used *AND* the PPS time has been initialized
        #    2. there is no PPS time, which will run off local system time
        if ((self.use_PPS_time == True and self.d_rxTime.isInitialized()) or (self.use_PPS_time == False)):
            for inIdx in range(num_input_items):
                fcM = self.fc/1e6
                fsM = self.fs/1e6 
                if self.writeWBIQFile == True:
                    dt = datetime.fromtimestamp(current_time // 1000000000)
                    in0[inIdx].tofile(f"{fcM}.{fsM}.{current_time}._cf32.sigmf-data")                    
                    metaWB = SigMFFile(
                        data_file=f"{fcM}.{fsM}.{current_time}._cf32.sigmf-data",
                        global_info = {
                            SigMFFile.DATATYPE_KEY: get_data_type_str(in0[inIdx]),
                            SigMFFile.SAMPLE_RATE_KEY: self.fs,
                            SigMFFile.AUTHOR_KEY: 'jane.doe@domain.org',
                            SigMFFile.DESCRIPTION_KEY: 'complex float32 debug file.',
                            SigMFFile.VERSION_KEY: "1.0.0",
                        }
                    )
                    metaWB.add_capture(0, metadata={
                        SigMFFile.FREQUENCY_KEY: self.fc,
                        SigMFFile.DATETIME_KEY: dt.isoformat()+'Z',
                    })
                if torch.cuda.is_available() == True and self.gpuDevice != 'cpu':
                    data = torch.from_numpy(in0[inIdx]).to('cuda'+':'+str(int(self.gpuDevice)))
                else:
                    data = torch.from_numpy(in0[inIdx])
                    self.gpuDevice = 'cpu'


                spectrogram = torchaudio.transforms.Spectrogram(
                              n_fft=self.nfft,
                              win_length=self.nfft,
                              hop_length=self.nfft,
                              window_fn=torch.blackman_window,
                              normalized=False,
                              center=False,
                              onesided=False,
                              power=2,
                              )
                if self.gpuDevice != 'cpu':
                    spectrogram.to('cuda'+':'+str(int(self.gpuDevice)))   
                    if self.gpuHalf:
                        spectrogram.half()
                else:
                    spectrogram.to('cpu')                                
                norm = lambda x: torch.linalg.norm(
                              x,
                              ord=float("inf"),
                              keepdim=True,
                              )      
                x = spectrogram(data)
                if self.gpuDevice != 'cpu':
                    x.to('cuda'+':'+str(int(self.gpuDevice)))                 
                x = x * (1 / norm(x.flatten()))
                x = torch.fft.fftshift(x,dim=0).flipud()
                x = 10*torch.log10(x+1e-12)

                if self.gpuDevice != 'cpu':
                    img_new = torch.zeros((self.nfft,self.nfft,3),device='cuda'+':'+str(int(self.gpuDevice)))
                    a = torch.tensor([[1,torch.max(torch.max(x))],[1,torch.min(torch.min(x))]],device='cuda'+':'+str(int(self.gpuDevice)))
                    b = torch.tensor([1,0],device='cuda'+':'+str(int(self.gpuDevice))).type(torch.float)
                    xx = torch.linalg.solve(a,b).to('cuda'+':'+str(int(self.gpuDevice))) 
                    intercept = xx[0]
                    slope = xx[1]
                    for j in range(3):
                        img_new[:,:,j] = (x*slope + intercept)
                    new_img_new = img_new.permute(-1,0,1).reshape(1,3,img_new.size(0),img_new.size(1)).to('cuda'+':'+str(int(self.gpuDevice)))
                    new_img_new = 1 - new_img_new                 
                else:
                    img_new = torch.zeros((self.nfft,self.nfft,3))    
                    a = torch.tensor([[1,torch.max(torch.max(x))],[1,torch.min(torch.min(x))]])
                    b = torch.tensor([1,0]).type(torch.float)
                    xx = torch.linalg.solve(a,b)
                    intercept = xx[0]
                    slope = xx[1]
                    for j in range(3):
                        img_new[:,:,j] = (x*slope + intercept)
                    new_img_new = img_new.permute(-1,0,1).reshape(1,3,img_new.size(0),img_new.size(1))
                    new_img_new = 1 - new_img_new   

                result = self.wb_model(new_img_new, imgsz=self.nfft, augment=self.augment,iou=self.iou, max_det=self.maxDet,agnostic_nms=self.agnosticNms, conf=self.conf, half=self.gpuHalf)

                plot_img = result[0].orig_img               
                if self.writeWBImages:
                    cv2.imwrite(f"{fcM}.{fsM}.{current_time}.inputImage.png",plot_img)
                if self.writeLabeledWBImages:
                    plot_img_l = result[0].plot()                
                    cv2.imwrite(f"{fcM}.{fsM}.{current_time}.YOLOlabeledImage.png",plot_img_l)                        
                startTime = np.uint64(current_time)
                endTime = np.uint64(startTime + int(((self.nfftSamples*1e9)/self.fs)))
                durationTime = endTime - startTime
                boxes_pmt = pmt.make_dict()
                boxes_pmt_dict = {}                
                detect_boxes = pmt.make_dict()
                detect_boxes_dict = {}                
                z = 0
                detect = False
                boxes_pmt_sum_dict = {}                
                for z, boxes_xyxy in enumerate(result[0].boxes.xyxy):
                    detect = True   
                    mod_nb = 'signal'
                    detect_dict = pmt.make_dict()
                    detectDict = {}   
                    mod_wb = result[0].names[int(result[0].boxes.cls[z].cpu().numpy())]  
                    if self.wb_detect_only:
                      mod_wb = 'signal'       
                    box_xyxy = boxes_xyxy.cpu().numpy()
                    box_xywh = result[0].boxes.xywh[z].cpu().numpy()
                    center_freq = ((float(self.fs)/2.0)-(float(box_xywh[1]/self.nfft)*float(self.fs))+self.fc)
                    top_freq = ((float(self.fs)/2.0)-((box_xyxy[1]/self.nfft)*float(self.fs))+self.fc)
                    bottom_freq = ((float(self.fs)/2.0)-((box_xyxy[3]/self.nfft)*float(self.fs))+self.fc)
                    bandwidth = top_freq - bottom_freq
                    start_sample = int(box_xyxy[0])*int(self.nfft)
                    end_sample = int(box_xyxy[2])*int(self.nfft)
                    duration = end_sample - start_sample                  
                    if start_sample > 0:
                        offset = (start_sample*1e9)/self.fs
                        length = (duration*1e9)/self.fs
                        length = np.uint64(length)
                        offset = np.uint64(offset)
                        start_time = offset + current_time
                        end_time = start_time + length
                    else:
                        start_time = current_time
                        length = (duration*1e9)/self.fs
                        length = np.uint64(length)    
                        end_time = start_time + length  
                    if self.writeNBIQFile or self.trainedNarrowbandModel != None:
                        usb = 0
                        lsb = 0
                        if top_freq > self.upper_limit:
                            top_freq = self.upper_limit
                            usb = top_freq - center_freq
                        if bottom_freq < self.lower_limit:
                            bottom_freq = self.lower_limit
                            lsb = bottom_freq - center_freq
                        if usb != 0 or lsb != 0:
                            if usb != 0 and lsb != 0:
                                bandwidth = min(lsb,usb)*2
                            else:
                                bandwidth = max(lsb,usb)*2            
                        if start_sample < 0:
                            start_sample=0
                        if end_sample >= self.nfft*self.nfft:
                            end_sample = (self.nfft*self.nfft) - 1  
                        if start_sample < end_sample:                      
                            num_samples = int(np.ceil(duration/(self.fs/np.ceil(bandwidth))))
                            truncate = False
                            gpu = False
                            if self.writeNBIQFile or self.trainedNarrowbandModel != None and num_samples > 0:
                                if self.trainedNarrowbandModel != None and num_samples >= 4096 and self.gpuDevice != 'cpu':
                                    truncate = True
                                    gpu = True
                                elif self.trainedNarrowbandModel != None and num_samples >= 4096 and self.gpuDevice == 'cpu':   
                                    truncate = True
                                    gpu = False 
                                elif self.trainedNarrowbandModel == None and num_samples >= 0 and self.gpuDevice != 'cpu' and self.writeNBIQFile:   
                                    truncate = False
                                    gpu = True                         
                                elif self.trainedNarrowbandModel == None and num_samples >= 0 and self.gpuDevice == 'cpu' and self.writeNBIQFile:
                                    truncate = False  
                                    gpu = False  
                                if num_samples >= 0 and (self.writeNBIQFile or self.trainedNarrowbandModel != None):
                                    new_data, osrate_new, gpu = self.baseband_downsample_complex_to_complex(data[start_sample:end_sample],self.fc, center_freq, self.fs,np.ceil(bandwidth),truncate)   
                                    
                                    
                            if truncate: #model_mod_rec takes in fixed size 4096         
                                if gpu and self.trainedNarrowbandModel != None:
                                    with torch.cuda.device('cuda'+':'+str(int(self.gpuDevice))):
                                        self.nb_model_mod_rec.to('cuda'+':'+str(int(self.gpuDevice)))
                                        self.nb_model_mod_rec.eval()
                                        if self.gpuHalf:
                                            self.nb_model_mod_rec.half()
                                        with torch.no_grad():
                                            data_mod = torch.stack((new_data.real,new_data.imag))
                                            nrm = torch.norm(data_mod,p=torch.inf,keepdim=True)
                                            data_mod = data_mod/nrm
                                            if self.gpuHalf:
                                                data_mod = data_mod.half()
                                            logits_mod = self.nb_model_mod_rec(data_mod.unsqueeze(0))
                                            preds_mod = torch.argmax(logits_mod, dim=1)
                                            mod_nb = self.nb_mod_rec_list[int(preds_mod.cpu().numpy())]
                                elif self.trainedNarrowbandModel != None and gpu != True:
                                    self.nb_model_mod_rec.eval()
                                    with torch.no_grad():
                                        data_mod = torch.from_numpy(new_data)
                                        data_mod = torch.stack((data_mod.real,data_mod.imag))
                                        nrm = torch.norm(data_mod,p=torch.inf,keepdim=True)
                                        data_mod = data_mod/nrm
                                        logits_mod = self.nb_model_mod_rec(data_mod.unsqueeze(0))
                                        preds_mod = torch.argmax(logits_mod, dim=1)
                                        mod_nb = self.nb_mod_rec_list[int(preds_mod.cpu().numpy())]                                                                                                                                
                            if self.writeNBIQFile and num_samples >= 0:
                                dt = datetime.fromtimestamp(start_time // 1000000000)
                                if gpu:
                                    new_data.cpu().numpy().tofile(f"{fcM}.{fsM}.{current_time}.{start_time}.{z}.{mod_nb}._cf32.sigmf-data")
                                        
                                    metaNB = SigMFFile(
                                        data_file=f"{fcM}.{fsM}.{current_time}.{start_time}.{z}.{mod_nb}._cf32.sigmf-data",
                                        global_info = {
                                            SigMFFile.DATATYPE_KEY: get_data_type_str(new_data.cpu().numpy()),
                                            SigMFFile.SAMPLE_RATE_KEY: osrate_new,
                                            SigMFFile.AUTHOR_KEY: 'jane.doe@domain.org',
                                            SigMFFile.DESCRIPTION_KEY: 'complex float32 debug file. '+mod_nb,
                                            SigMFFile.VERSION_KEY: "1.0.0",
                                        }
                                    )                                    
                                else:
                                    new_data.tofile(f"{fcM}.{fsM}.{current_time}.{start_time}.{z}.{mod_nb}._cf32.sigmf-data")     
                                    metaNB = SigMFFile(
                                        data_file=f"{fcM}.{fsM}.{current_time}.{start_time}.{z}.{mod_nb}._cf32.sigmf-data",
                                        global_info = {
                                            SigMFFile.DATATYPE_KEY: get_data_type_str(new_data),
                                            SigMFFile.SAMPLE_RATE_KEY: osrate_new,
                                            SigMFFile.AUTHOR_KEY: 'jane.doe@domain.org',
                                            SigMFFile.DESCRIPTION_KEY: 'complex float32 debug file. '+mod_nb,
                                            SigMFFile.VERSION_KEY: "1.0.0",
                                        }
                                    )
                                metaNB.add_capture(0, metadata={
                                    SigMFFile.FREQUENCY_KEY: osrate_new,
                                    SigMFFile.DATETIME_KEY: dt.isoformat()+'Z',
                                }) 
                                metaNB.tofile(f"{fcM}.{fsM}.{current_time}.{start_time}.{z}.{mod_nb}._cf32.sigmf-meta")                                                            
                    if self.writeWBIQFile == True:
                        metaWB.add_annotation(start_sample, duration, metadata = {
                            SigMFFile.FLO_KEY: bottom_freq,
                            SigMFFile.FHI_KEY: top_freq,
                            SigMFFile.COMMENT_KEY: 'wb modulation: '+mod_wb+' nb modulation: '+mod_nb,
                        })                  
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('box_xyxy'),pmt.to_pmt(box_xyxy))   
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('box_xywh'),pmt.to_pmt(box_xywh)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('narrowband_modulation'),pmt.to_pmt(mod_nb))
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('wideband_modulation'),pmt.to_pmt(mod_wb)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('center_freq'),pmt.from_float(center_freq))
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('top_freq'),pmt.from_float(top_freq)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('bottom_freq'),pmt.from_float(bottom_freq)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('bandwidth'),pmt.from_float(bandwidth)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('start_sample'),pmt.from_long(start_sample)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('end_sample'),pmt.from_long(end_sample))  
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('duration'),pmt.from_long(duration)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('start_time'),pmt.from_uint64(start_time))      
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('end_time'),pmt.from_uint64(end_time))  
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('length'),pmt.from_uint64(length))           
                    boxes_pmt = pmt.dict_add(boxes_pmt, pmt.intern(str(z)),detect_dict)
                    if self.detectJson:
                        detectDict['box_xyxy'] = [float(i) for i in list(box_xyxy)] 
                        detectDict['box_xywh'] = [float(i) for i in list(box_xywh)]
                        detectDict['center_freq'] = float(center_freq)
                        detectDict['modulation_wb'] = mod_wb
                        detectDict['modulation_nb'] = mod_nb
                        detectDict['top_freq'] = float(top_freq) 
                        detectDict['bottom_freq'] = float(bottom_freq)
                        detectDict['bandwidth'] = float(bandwidth)
                        detectDict['start_sample'] = int(start_sample)
                        detectDict['end_sample'] = int(end_sample)
                        detectDict['duration'] = int(duration)
                        detectDict['start_time'] = int(start_time)
                        detectDict['end_time'] = int(end_time) 
                        detectDict['length'] = int(length) 
                        boxes_pmt_sum_dict[str(z)] = detectDict 
                if self.detectJson:                
                    boxes_pmt_dict['detects'] = boxes_pmt_sum_dict
                    boxes_pmt_dict['detect_count'] = int(z+1)
                    boxes_pmt_dict['detect'] = detect
                    detect_boxes_dict['boxes_pmt'] = boxes_pmt_dict  
                    detect_boxes_dict['fcM'] = float(fcM) 
                    detect_boxes_dict['fsM'] = float(fsM) 
                    detect_boxes_dict['startTime'] = int(startTime)      
                    detect_boxes_dict['endTime'] = int(endTime) 
                    detect_boxes_dict['durationTime'] = int(durationTime)       
                    detect_boxes_dict['FFTSize'] = int(self.nfft)      
                    detect_boxes_dict['FFTxFFT'] = int(self.nfftSamples)
                    detect_boxes_dict['trainedWidebandModel'] = str(self.wb_model_path)
                    detect_boxes_dict['trainedNarrowbandModel'] = str(self.trainedNarrowbandModel)
                    detect_boxes_dict['half'] = str(self.gpuHalf)
                    detect_boxes_dict['max_det'] = str(self.maxDet)
                    detect_boxes_dict['iou'] = str(self.iou)
                    detect_boxes_dict['conf'] = str(self.conf)
                    detect_boxes_dict['augment'] = str(self.augment)
                    detect_boxes_dict['agnostic_nms'] = str(self.agnosticNms)
                    detect_boxes_dict['ultralytics'] = str(ultralytics.__version__)
                    detect_boxes_dict['torch'] = str(torch.__version__)
                    detect_boxes_dict['torchsig'] = str(torchsig.__version__)
                    detect_boxes_dict['torchaudio'] = str(torchaudio.__version__)
                    detect_boxes_dict['cv2'] = str(cv2.__version__)
                    detect_boxes_dict['sigmf'] = str(sigmf.__version__)  
                    with open(f"{fcM}.{fsM}.{current_time}.detect.json", "w",encoding='utf-8') as outfile:
                        json.dump(detect_boxes_dict, outfile)                   
                boxes_pmt = pmt.dict_add(boxes_pmt, pmt.intern('detect_count'),pmt.from_long(z+1))
                boxes_pmt = pmt.dict_add(boxes_pmt, pmt.intern('detect'),pmt.from_bool(detect))  
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('boxes_pmt'),boxes_pmt)
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('plot_img'),pmt.to_pmt(plot_img))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('fcM'),pmt.from_float(fcM))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('fsM'),pmt.from_float(fsM))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('startTime'),pmt.from_uint64(startTime))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('endTime'),pmt.from_uint64(endTime))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('durationTime'),pmt.from_uint64(durationTime))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('FFTSize'),pmt.from_long(int(self.nfft)))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('FFTxFFT'),pmt.from_long(int(self.nfftSamples)))



                self.message_port_pub(pmt.intern(self.portName1), detect_boxes)
                if self.writeWBIQFile == True:
                    metaWB.tofile(f"{fcM}.{fsM}.{current_time}._cf32.sigmf-data")
                       
 
        return num_input_items
