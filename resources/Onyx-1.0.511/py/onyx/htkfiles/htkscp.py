###########################################################################
#
# File:         htkscp.py
# Date:         Wed 5 Nov 2008 13:59
# Author:       Ken Basye
# Description:  Tools to read multiple HTK audio files as specified by one scp file
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008 The Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
###########################################################################

"""
    Tools to read multiple HTK audio files as specified by one scp file

    HTK SCP files are just lists of filenames, one per line.  The ScpProcessor takes these lines, opens
    and reads the file, and generates an HTKAudioBlockEvent for each file.
    
    >>> scpProc = ScpProcessor(prefix_subst=('/Users/kbasye1/work/R1/mfcc//ind_trn/adg0_4',
    ...                                       onyx.home + '/data/htk_r1_adg0_4'))


    BlockToFeatures is a simple processor which takes HTKAudioBlockEvents and
    generates feature vectors as Numpy arrays.
    
    >>> result = []
    >>> b2f = BlockToFeatures(result.append)
    >>> scpProc.set_sendee(b2f.process)

    >>> module_dir, module_name = os.path.split(__file__)

    >>> scp_file_name = os.path.join(module_dir, "ind_trn109.scp")
    >>> with open(scp_file_name) as scp_file:
    ...     limited_iter = islice(scp_file, 10)   # only process the first 10 lines 
    ...     source = IteratorSource(limited_iter, scpProc.process)
    ...     source.start()
    ...     source.wait_iter_stopped()
    ...     source.stop(flush=True)
    ...     source.wait()
    ...     source.done()
    True

    >>> len(result)
    3294
    >>> print result[0]
    [ -1.57193413e+01  -5.69683504e+00  -1.42094164e+01  -1.09782257e+01
      -9.74443531e+00  -6.75594759e+00  -2.05153537e+00  -2.27887034e+00
      -8.51778793e+00  -1.37430739e+00   7.37931538e+00   4.42070436e+00
      -1.05129251e+01  -3.37096214e-01   8.85067165e-01   1.51379669e+00
       8.33324611e-01  -4.58665192e-01   8.61221790e-01   5.57038009e-01
       5.02852559e-01  -6.86278343e-01  -1.18701637e+00  -3.33604860e+00
      -4.29879606e-01   0.00000000e+00  -1.03502944e-01  -9.11712088e-03
       2.38455529e-03   3.19893450e-01   2.66819388e-01   6.52765259e-02
       1.64075077e-01   3.10689837e-01   7.19753325e-01   4.09678102e-01
       5.58297873e-01   3.91171537e-02   0.00000000e+00]

    >>> data = load_scp_file(scp_file_name, max_lines=20,
    ...                      prefix_subst=('/Users/kbasye1/work/R1/mfcc//ind_trn/adg0_4',
    ...                                     onyx.home + '/data/htk_r1_adg0_4'))
    >>> print len(data)
    20
    >>> fname0 = data[0][0]
    >>> os.path.basename(fname0)
    'adg0_4_sr009.mfc'

"""

from __future__ import with_statement
import onyx
import os
from itertools import islice
from onyx.htkfiles.htkaudio import read_htk_audio_file
from onyx.util.streamprocess import ProcessorBase
from onyx.dataflow.source import IteratorSource

class HTKAudioBlockEvent(object):
    """
    Event corresponding to a block of audio from an HTK source.

    Generated, e.g., by reading through an scp file, see ScpProcessor.
    """
    def __init__(self, record_filename, meta_data, data):
        self.record_filename = record_filename
        self.meta_data = meta_data
        self.data = data
        
    def __repr__(self):
        return "HTKAudioBlockEvent from %s, meta_data = %s" % (self.record_filename,
                                                               self.meta_data)


class BlockToFeatures(ProcessorBase):
    """
    Transform HTKAudioBlockEvents into a stream of raw features.

    Events sent: Numpy arrays of features
    """
    def __init__(self, sendee=None, sending=True):
        """
        Transform HTKAudioBlockEvents into a stream of raw features.
        
        sendee is a function to call with our output events.  
        """
        super(BlockToFeatures, self).__init__(sendee, sending=sending)

    def process(self, block):
        """
        Process an HTKAudioBlockEvent, generate feature vectors and push them on.
        """        
        for vec in block.data:
            self.send(vec)
    

class ScpProcessor(ProcessorBase):
    """
    Generator for blocks of HTK audio data from a stream of scp lines.
    
    Event types sent:  HTKAudioBlockEvent
    """

    def __init__(self, sendee=None, sending=True, prefix_subst=None):
        """
        sendee is a function to call with our output events.  prefix_subst, if provided, is a tuple
        of two strings.  Any lines in the file which have the first string as a prefix will have
        that prefix replaced by the second string.
        """
        super(ScpProcessor, self).__init__(sendee, sending=sending)
        if prefix_subst is not None:
            if ((not hasattr(prefix_subst, '__len__')) or
                len(prefix_subst) != 2 or
                (not isinstance(prefix_subst[0], str)) or 
                (not isinstance(prefix_subst[1], str))):
                raise ValueError("prefix subst must be a tuple of two strings, got %s" % (prefix_subst,))
        self._prefix_subst = prefix_subst
        
    def process(self, line):
        """
        Process a line from an SCP file, generate an MlfBlock Event from the
        filename, and call the sendee with it
        """        
        # print line
        fname = line.strip()
        # We silently skip any blank or whitespace-only lines
        if len(fname) == 0:
            return
        if self._prefix_subst is not None and fname.startswith(self._prefix_subst[0]):
            fname = fname.replace(self._prefix_subst[0], self._prefix_subst[1], 1)
        with open(fname, 'rb') as f:
            (data, (kind, qualifiers), samp_period) = read_htk_audio_file(f)
        self.send(HTKAudioBlockEvent(fname, (kind, qualifiers, samp_period), data))

def load_scp_file(scp_file_name, max_lines=None, prefix_subst=None):
    """
    Open an HTK scp file, then open all the audio files it refers to.
    """
    all_data = []

    def extract(evt):
        assert isinstance(evt, HTKAudioBlockEvent)
        all_data.append((evt.record_filename, evt.data))

    proc = ScpProcessor(sendee=extract, prefix_subst=prefix_subst)
    with open(scp_file_name) as scp_file:
        for i, line in enumerate(scp_file):
            if max_lines and i >= max_lines:
                break
            proc.process(line)
    return all_data
                
                
if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



