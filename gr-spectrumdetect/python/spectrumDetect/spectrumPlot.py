#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-spectrumDetect author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
from PyQt5 import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib.path import Path 
from matplotlib.patches import PathPatch
import pmt

class spectrumPlot(gr.sync_block, QWidget):
    """
    docstring for block spectrumPlot
    """
    def __init__(self,save,lbl):
        gr.sync_block.__init__(self,
            name="spectrumPlot",
            in_sig=None,
            out_sig=None)       

        QWidget.__init__(self)   
        self.save = save       
        self.figure = plt.figure(figsize=(40,40))
        self.ax = plt.subplot()
        self.lbl = lbl
        self.figure.suptitle(self.lbl, fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Time Bins")
        self.ax.set_ylabel("FFT Bins")
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]
        self.portName1 = "detect_pmt"
        self.message_port_register_in(pmt.intern(self.portName1))
        self.set_msg_handler(pmt.intern(self.portName1), self.plot)
       

    def plot(self, msg):
        self.ax.cla()
        detect_boxes = pmt.to_python(msg)
        nfft = detect_boxes['FFTSize'] 
        plot_img = detect_boxes['plot_img'].reshape(nfft,nfft,3)
        cfreqMHz = detect_boxes['fcM']
        plotFreqBW = detect_boxes['fsM']
        duration = str(detect_boxes['durationTime'])
        FFTSize = detect_boxes['FFTSize']
        nfftSamples = detect_boxes['FFTxFFT']
        if detect_boxes['boxes_pmt']['detect']:
            for cnt in range(detect_boxes['boxes_pmt']['detect_count']):
                plot_box_xyxy = detect_boxes['boxes_pmt'][str(cnt)]['box_xyxy']
                plot_box_xywh = detect_boxes['boxes_pmt'][str(cnt)]['box_xywh']
                verts = [
                    (plot_box_xyxy[0], plot_box_xyxy[1]+plot_box_xywh[3]),
                    (plot_box_xyxy[0], plot_box_xyxy[1]),
                    (plot_box_xyxy[0]+plot_box_xywh[2], plot_box_xyxy[1]),
                    (plot_box_xyxy[2], plot_box_xyxy[3]),
                    (0., 0.),
                ]
                path = Path(verts, self.codes)
                patch = PathPatch(path, facecolor='none', lw=2, edgecolor='red', alpha=.3)
                self.ax.add_patch(patch)
                self.ax.text(plot_box_xyxy[0],plot_box_xyxy[1], 'st: '+str(detect_boxes['boxes_pmt'][str(cnt)]['start_time'])+'(ns)', color='cyan',fontsize=10)
                self.ax.text(plot_box_xyxy[0],plot_box_xyxy[1]+(0.5*plot_box_xywh[3]), 'fc: '+'%.6s' % str(detect_boxes['boxes_pmt'][str(cnt)]['center_freq']/1e6)+'(MHz)', color='yellow',fontsize=10)
                if str(detect_boxes['boxes_pmt'][str(cnt)]['wideband_modulation']) != 'signal':
                    self.ax.text(plot_box_xyxy[0],plot_box_xyxy[1]+(0.7*plot_box_xywh[3]), 'wb_mod: '+ str(detect_boxes['boxes_pmt'][str(cnt)]['wideband_modulation']), color='chartreuse',fontsize=10)
                if str(detect_boxes['boxes_pmt'][str(cnt)]['narrowband_modulation']) != 'signal':    
                    self.ax.text(plot_box_xyxy[0],plot_box_xyxy[1]+(0.9*plot_box_xywh[3]), 'nb_mod: '+ str(detect_boxes['boxes_pmt'][str(cnt)]['narrowband_modulation']), color='lawngreen',fontsize=10)

       
               
        self.ax.imshow(plot_img)
        self.ax.set_xlabel("Duration: "+str(duration)+' (ns), Start Time (ns): '+str(detect_boxes['startTime'])+' End Time (ns): '+str(detect_boxes['endTime']))
        self.ax.set_ylabel('Bandwidth: '+str(plotFreqBW)+'(MHz), Center Frequency: '+str(cfreqMHz)+'(MHz)')
        self.ax.set_xticks([0,(nfft/2)-1,nfft-1],[detect_boxes['startTime'],np.uint64(detect_boxes['startTime'])+np.uint64(np.uint64((detect_boxes['durationTime'])/2)),detect_boxes['endTime']])
        self.ax.set_yticks([0,(nfft/2)-1,nfft-1],[cfreqMHz+plotFreqBW/2.0,cfreqMHz,cfreqMHz-plotFreqBW/2.0])
        if self.save:
            self.figure.savefig(str(detect_boxes['startTime'])+'.png')
        self.canvas.draw()
