#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-spectrumDetect author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
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

class specDetect(gr.sync_block):

    def __init__(self, centerFrequency, sampleRate, vectorSize, nfft, trainedModel, augment, iou, save, max_det,debug_sigMF):
        gr.sync_block.__init__(self,
            name="specDetect",
            in_sig=[(np.complex64,vectorSize)],
            out_sig=None)
        self.fc = centerFrequency    
        self.fs = sampleRate            
        self.nfft = nfft
        self.nfftSamples = vectorSize
        self.model_path = trainedModel    
        self.model = YOLO(self.model_path)
        self.portName1 = "detect_pmt"
        self.message_port_register_out(pmt.intern(self.portName1))
        self.d_rxTime = rxTime(vlen=nfft)
        self.augment = augment
        self.iou = iou
        self.save = save
        self.max_det = max_det
        self.use_PPS_time = False
        self.debug_sigMF = debug_sigMF
                
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
                if self.debug_sigMF == True:
                    dt = datetime.fromtimestamp(current_time // 1000000000)
                    in0[inIdx].tofile('./debugData_'+str(current_time)+'_cf32.sigmf-data')
                    meta = SigMFFile(
                        data_file='debugData_'+str(current_time)+'_cf32.sigmf-data',
                        global_info = {
                            SigMFFile.DATATYPE_KEY: get_data_type_str(in0[inIdx]),
                            SigMFFile.SAMPLE_RATE_KEY: self.fs,
                            SigMFFile.AUTHOR_KEY: 'jane.doe@domain.org',
                            SigMFFile.DESCRIPTION_KEY: 'complex float32 debug file.',
                            SigMFFile.VERSION_KEY: "1.0.0",
                        }
                    )
                    meta.add_capture(0, metadata={
                        SigMFFile.FREQUENCY_KEY: self.fc,
                        SigMFFile.DATETIME_KEY: dt.isoformat()+'Z',
                    })
                if torch.cuda.is_available() == True:
                    torch.set_default_device('cuda')
                    data = torch.from_numpy(in0[inIdx]).cuda()
                else:
                    data = torch.from_numpy(in0[inIdx])


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
                norm = lambda x: torch.linalg.norm(
                              x,
                              ord=float("inf"),
                              keepdim=True,
                              )      
                x = spectrogram(data)
                x = x * (1 / norm(x.flatten()))
                x = torch.fft.fftshift(x,dim=0).flipud()
                x = 10*torch.log10(x+1e-12)

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

                result = self.model(new_img_new, imgsz=self.nfft, save=self.save, augment=self.augment,iou=self.iou, max_det=self.max_det)

                plot_img = result[0].orig_img
                fcM = self.fc/1e6
                fsM = self.fs/1e6
                startTime = np.uint64(current_time)
                endTime = np.uint64(startTime + int(((self.nfftSamples*1e9)/self.fs)))
                durationTime = endTime - startTime
                boxes_pmt = pmt.make_dict()
                detect_boxes = pmt.make_dict()
                z = 0
                detect = False
                for z, boxes_xyxy in enumerate(result[0].boxes.xyxy):
                    detect = True   
                    detect_dict = pmt.make_dict()           
                    box_xyxy = boxes_xyxy.cpu().numpy()
                    box_xywh = result[0].boxes.xywh[z].cpu().numpy()
                    center_freq = ((float(self.fs)/2.0)-(float(box_xywh[1]/self.nfft)*float(self.fs))+self.fc)
                    top_freq = ((float(self.fs)/2.0)-((box_xyxy[1]/self.nfft)*float(self.fs))+self.fc)
                    bottom_freq = ((float(self.fs)/2.0)-((box_xyxy[3]/self.nfft)*float(self.fs))+self.fc)
                    bandwidth = top_freq - bottom_freq
                    start_sample = int(box_xyxy[0])*int(self.nfft)
                    end_sample = int(box_xyxy[2])*int(self.nfft)
                    duration = end_sample - start_sample
                    if self.debug_sigMF == True:
                        meta.add_annotation(start_sample, duration, metadata = {
                            SigMFFile.FLO_KEY: bottom_freq,
                            SigMFFile.FHI_KEY: top_freq,
                            SigMFFile.COMMENT_KEY: 'signal',
                        })                    
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
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('box_xyxy'),pmt.to_pmt(box_xyxy))   
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('box_xywh'),pmt.to_pmt(box_xywh))  
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
                if self.debug_sigMF == True:
                    meta.tofile('./debugData_'+str(current_time)+'_cf32.sigmf-meta')
 
        return num_input_items
