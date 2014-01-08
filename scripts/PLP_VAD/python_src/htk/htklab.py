# Copyright (c) 2007 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""Read and write HTK transcription files.

This module reads and writes the lab files used by HTK
"""

__author__ = "Nassos Katsamanis <nkatsam@sipi.usc.edu>"
__version__ = "$Revision $"

from struct import unpack, pack
import numpy

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0000100 # has energy
_N = 0000200 # absolute energy supressed
_D = 0000400 # has delta coefficients
_A = 0001000 # has acceleration (delta-delta) coefficients
_C = 0002000 # is compressed
_Z = 0004000 # has zero mean static coefficients
_K = 0010000 # has CRC checksum
_O = 0020000 # has 0th cepstral coefficient
_V = 0040000 # has VQ data
_T = 0100000 # has third differential coefficients

#def open(f, mode=None, veclen=13):
#    """Open an HTK format feature file for reading or writing.
#    The mode parameter is 'rb' (reading) or 'wb' (writing)."""
#    if mode is None:
#        if hasattr(f, 'mode'):
#            mode = f.mode
#        else:
#            mode = 'rb'
#    if mode in ('r', 'rb'):
#        return HTKlab_read(f) # veclen is ignored since it's in the file
#    elif mode in ('w', 'wb'):
#        return HTKlab_write(f, veclen)
#    else:
#        raise Exception, "mode must be 'r', 'rb', 'w', or 'wb'"

class HTKlab_read(object):
    "Read HTK format feature files"
    def __init__(self, filename=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(filename)

    def __iter__(self):
        self.fh.seek(12,0)
        return self

    def open(self, filename):
        self.filename = filename
        self.fh = file(filename, "rb")
        self.readheader()

    def readheader(self):
        self.fh.seek(0,0)
        spam = self.fh.read(12)
        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = \
                       unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize / 2
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = numpy.fromfile(self.fh, 'f', self.veclen)
                self.B = numpy.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'    
            self.veclen = self.sampSize / 4
        self.hdrlen = self.fh.tell()

    def seek(self, idx):
        self.fh.seek(self.hdrlen + idx * self.sampSize, 0)

    def next(self):
        vec = numpy.fromfile(self.fh, self.dtype, self.veclen)
        if len(vec) == 0:
            raise StopIteration
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self):
        return self.next()

    def getall(self):
        self.seek(0)
        data = numpy.fromfile(self.fh, self.dtype)
        if self.parmKind & _K: # Remove and ignore checksum
            data = data[:-1]
        data = data.reshape(len(data)/self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

class HTKlab_write(object):
    "Write HTK format feature files"
    def __init__(self, filename=None,
                 samp_period=100000, labels=None,
                 ):
        self.samp_period = samp_period
        if labels == None:
            raise ValueError('Label names should be defined')
        else:
            self.labels = labels
        
        if (filename != None):
            self.lab_open(filename)

    def __del__(self):
        self.fh.close()

    def lab_open(self, filename):
        self.filename = filename
        self.fh = open(filename, "w")
        
    def write(self, ind_vector):
        '''
        Convert indicator vector to segment start and end times 
        and write to file
        '''
        start_time = 0
        cur_class = None
        end_time = 0
        n_times = len(ind_vector)
        wrote_last_label = False
        
        for tm in range(n_times):
            cur_time = tm*self.samp_period
            l_id = ind_vector[tm]
            wrote_last_label = False
            if l_id != cur_class:
                if end_time > start_time:
                    assert(cur_class!=None)
                    self.fh.write("{start_time} {end_time} {label}\n".format(
                                                                           start_time=int(start_time),
                                                                           end_time=int(end_time),
                                                                           label=self.labels[cur_class]))
                    start_time = cur_time
                    wrote_last_label = True
                cur_class = l_id
                
            end_time = cur_time + self.samp_period
        
        if not wrote_last_label:
            if end_time > start_time:
                assert(cur_class!=None)
                self.fh.write("{start_time} {end_time} {label}\n".format(
                                                                         start_time=int(start_time),
                                                                         end_time=int(end_time),
                                                                         label=self.labels[cur_class]))
