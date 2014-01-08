###########################################################################
#
# File:         test0.py
# Date:         Thu 6 Nov 2008 15:18
# Author:       Ken Basye
# Description:  A first test of the Byrne R1 recipe for monophone testing - see byrne.README
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
    >>> module_dir, module_name = os.path.split(__file__)

    >>> scp_file_name = os.path.join(module_dir, "ind_trn109_short.scp")
    >>> mlf_file_name = os.path.join(module_dir, "mono_short.mlf")

    >>> mlf_file = open(mlf_file_name)
    >>> scp_file = open(scp_file_name)
    >>> labeled_data = []

    >>> prefix_subst=('/Users/kbasye1/work/R1/mfcc39//ind_trn/adg0_4', onyx.home + '/data/htk_r1_adg0_4')
    >>> process_htk_files(mlf_file, scp_file, labeled_data.append, scp_prefix_subst=prefix_subst)

    >>> scp_file.close()
    >>> mlf_file.close()

    >>> len(labeled_data)
    40

    >>> print [evt.name for evt in labeled_data][0:5]
    ['adg0_4_sr009', 'adg0_4_sr049', 'adg0_4_sr089', 'adg0_4_sr129', 'adg0_4_sr169']

"""

from __future__ import with_statement, division
import os.path
import sys
import itertools
import contextlib
import time

import onyx
from onyx.am.modelmgr import GmmMgr
from onyx.am.hmm_mgr import HmmMgr, write_acoustic_model, read_acoustic_model
from onyx.am.bwtrainer import TrainingGraph
from onyx.util.debugprint import dcheck, DebugPrint
from onyx.graph.graphtools import SetGraphBuilder, FrozenGraph
from onyx.util.streamprocess import ProcessorBase
from onyx.htkfiles.mlfprocess import MLFProcessor, MLFBlockProcessor, MLFBlockEvent
from onyx.htkfiles.htkscp import ScpProcessor, HTKAudioBlockEvent
from onyx.dataflow.source import IteratorSource
from onyx.dataflow.join import SynchronizingSequenceJoin


class LabeledDataEvent(object):
    """
    Event corresponding to a paired transcript and block of audio.

    Generated, e.g., by an HtkLabeledDataProcessor.
    """
    def __init__(self, name, labels, data, meta_data):
        self.name = name
        self.labels = labels
        self.data = data
        self.meta_data = meta_data
        
    def __repr__(self):
        return "LabeledDataEvent %s with %d labels and %d frames; meta_data = %s" % (self.name,
                                                                                     len(self.labels),
                                                                                     self.data.shape[0],
                                                                                     self.meta_data)
    
class HtkLabeledDataProcessor(ProcessorBase):
    """
    Build packets of labeled data from HTK files.

    Event types sent:  LabeledDataEvent
    """

    def __init__(self, sendee=None, sending=True):
        """
        sendee is a function to call with our output events.
        """
        super(HtkLabeledDataProcessor, self).__init__(sendee, sending=sending)
        
    def process(self, pair):
        """
        Process a pair of events, one from an MLFBlockProcessor and the other
        from an ScpProcesser and perhaps generate a LabeledDataEvent and call
        the sendee with it
        """
        mlf_block, audio_block = pair
        if not isinstance(mlf_block, MLFBlockEvent):
            raise ValueError("expected an event of type MLFBlockEvent, got %s" % (type(mlf_block)))
        if not isinstance(audio_block, HTKAudioBlockEvent):
            raise ValueError("expected an event of type HTKAudioBlockEvent, got %s" % (type(audio_block)))

        mlf_name = os.path.basename(mlf_block.record_filename)
        audio_name = os.path.basename(audio_block.record_filename)

        mlf_base, dummy = os.path.splitext(mlf_name)
        audio_base, dummy = os.path.splitext(audio_name)

        if mlf_base == audio_base:
            out_event = LabeledDataEvent(mlf_base, mlf_block.labels, audio_block.data, audio_block.meta_data)
            self.send(out_event)
        else:
            print("Mismatched event pair ignored: %s %s%" % (mlf_base, audio_base))



def process_htk_files(mlf_file, scp_file, tail_process_function, scp_prefix_subst=None):
    """
    Read an HTK MLF (Master Label File) file and a corresponding scp file, call
    tail_process_function with results in the form of LabeledDataEvents.

    mlf_file and scp_file should be open file handles; tail_process_function
    should be a callable taking one argument.  scp_prefix_subst, if it is not
    None, will be passed to ScpProcessor to be used in converting filenames in
    the scp_file.

    We're going to set up the following dataflow network:

    label_source ==> line_proc ==> block_proc ==\
                                                  join  ==> ld_proc
    audio_source ==> audio_proc ================/

    """
    def handler(label_evt):
        return label_evt.word_label

    ld_proc = HtkLabeledDataProcessor(tail_process_function)
    join = SynchronizingSequenceJoin(ld_proc.process)

    # We need to make sure we call join.get_process_function in the right
    # order here, since the ld_proc is expecting pairs of (label_block, audio_block)
    block_proc = MLFBlockProcessor(handler, sendee=join.get_process_function())
    line_proc = MLFProcessor(block_proc.process)

    audio_proc = ScpProcessor(sendee=join.get_process_function(), prefix_subst=scp_prefix_subst)

    label_source = IteratorSource(mlf_file, sendee=line_proc.process)
    audio_source = IteratorSource(scp_file, sendee=audio_proc.process)

    label_source.start()
    audio_source.start()

    while not (label_source.is_iter_stopped and audio_source.is_iter_stopped):
        time.sleep(1/64)

    label_source.stop(flush=True)
    audio_source.stop(flush=True)
    label_source.wait()
    audio_source.wait()
    label_source.done()
    audio_source.done()
    



def labels_to_lattice(labels):
    """
    From a sequence of labels, build a linear lattice with labels on the arcs.
    The result will have node labels which are guaranteed to be unique.

    >>> label_dict = {'A': 1, 'B': 4, 'C': 9}
    >>> labels_to_lattice(('A', 'B', 'C'))
    FrozenGraph(GraphTables(((0, 1, 2, 3), (0, 1, 2), (1, 2, 3), ('A', 'B', 'C'))))
    """
    gb = SetGraphBuilder()
    counter = itertools.count()
    start = gb.add_node(counter.next())
    for l in labels:
        end = gb.add_node(counter.next())
        gb.add_arc(start, end, l)
        start = end
    return FrozenGraph(gb)

def build_model_lattice(label_lattice, model_dict, epsilon_index):
    """
    From a lattice with labels on the arcs and a dict mapping labels to model
    indices, build a lattice with (node-index, model index) pairs on the nodes,
    usable for constructing a TrainingGraph.

    The resulting lattice may have new epsilon nodes as the new start and end
    nodes; these will be given epsilon_index as their model indices.  Note that
    this function requires that label_lattice have unique labels on nodes.  XXX
    maybe do this node-labeling ourselves here?

    >>> label_dict = {'A': 1, 'B': 4, 'C': 9}
    >>> lat = labels_to_lattice(('A', 'B', 'C'))
    >>> lat
    FrozenGraph(GraphTables(((0, 1, 2, 3), (0, 1, 2), (1, 2, 3), ('A', 'B', 'C'))))

    >>> result = build_model_lattice(lat, label_dict, 15)
    >>> print result
    FrozenGraph(GraphTables((((0, 1), (1, 4), (2, 9)), (0, 1), (1, 2), (None, None))))

    # >>> result.dot_display()

    """
    if not label_lattice.is_lattice() or label_lattice.has_self_loop():
        raise ValueError("label_lattice is not a lattice or has a self loop")

    counter = itertools.count()
    # we need our node labels to be pairs of ints in which the first int is
    # unique and the second is the index of the model from the callers
    # label_dict
    def model_node_labeler(pred_node_label, arc_label, succ_node_label):
        if not model_dict.has_key(arc_label):
            raise KeyError("Failed on lookup of label %s" % (arc_label))
        model_index = model_dict[arc_label]
        return (counter.next(), model_index) 

    def empty_arc_labeler(in_arc_label, node_label, out_arc_label):
        return None
    
    line_graph = label_lattice.get_line_graph(model_node_labeler, empty_arc_labeler)
    starts, ends  = line_graph.get_terminals()
    num_starts = len(starts)
    num_ends = len(ends)
    # If we started with a lattice, the line graph must have some terminals
    assert num_starts >= 1 and num_ends >= 1

    start_labels = (line_graph.get_label(node_id) for node_id in starts)
    end_labels = (line_graph.get_label(node_id) for node_id in ends)
    gb = SetGraphBuilder(line_graph)

    # Tie terminals together with epsilons if necessary
    if num_starts > 1:
        new_start_label = gb.add_node((counter.next(), epsilon_index))
        for node_label in start_labels:
            gb.add_arc(new_start_label, node_label)

    if num_ends > 1:
        new_end_label = gb.add_node((counter.next(), epsilon_index))
        for node_label in end_labels:
            gb.new_arc(node_label, new_end_label)

    return FrozenGraph(gb)


def train_model(in_filename, mlf_filename, scp_filename, out_filename, scp_prefix_subst=None):
    # Read models and create managers
    with open(in_filename) as f:
        model_dict, gmm_mgr, hmm_mgr = read_acoustic_model(f, log_domain=True)

    # Some tricky business to make sure we have an epsilon Hmm to use later
    assert hmm_mgr.input_arity_set == hmm_mgr.output_arity_set
    assert len(hmm_mgr.input_arity_set) == 1
    in_out_arity = hmm_mgr.input_arity_set.pop()
    epsilon_index = hmm_mgr.add_epsilon_model(gmm_mgr, in_out_arity, log_domain=True)

    # Initialize managers for training
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()

    utterance_num = [1]
    
    def train_one_utterance(labeled_data_event):
        print("Beginning training on utterance %d (%s) (%d frames)" % (utterance_num[0], labeled_data_event.name,
                                                                  labeled_data_event.data.shape[0]))
        sys.stdout.flush()
        label_lattice = labels_to_lattice(labeled_data_event.labels)
        # Build model lattice
        model_lattice = build_model_lattice(label_lattice, model_dict, epsilon_index)
        # Build training graph      
        training_graph = TrainingGraph(model_lattice, hmm_mgr, dict())
        # Train on data
        
        if gmm_mgr.get_adaptation_state() != "INITIALIZING":
            gmm_mgr.set_adaptation_state("INITIALIZING")
            hmm_mgr.set_adaptation_state("INITIALIZING")
        training_graph.begin_training()
        gmm_mgr.set_adaptation_state("ACCUMULATING")
        hmm_mgr.set_adaptation_state("ACCUMULATING")

        training_graph.train_one_sequence(labeled_data_event.data)
        training_graph.end_training()
        print("Finished training on utterance %d (%s)" % (utterance_num[0], labeled_data_event.name))
        sys.stdout.flush()
        utterance_num[0] += 1

    # Do processing:
    with contextlib.nested(open(mlf_filename), open(scp_filename)) as (mlf_file, scp_file):
        process_htk_files(mlf_file, scp_file, train_one_utterance, scp_prefix_subst)

    # Finalize training
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")

    # Write out models
    with open(out_filename, "w") as f:
        write_acoustic_model(model_dict, gmm_mgr, hmm_mgr, f)


def test0(out_filename):
    in_filename = 'hmm0_r1.am'
    scp_filename = 'ind_trn109_vshort.scp'
    mlf_filename = 'mono_vshort.mlf'
    prefix_subst=('/Users/kbasye1/work/R1/mfcc39//ind_trn/adg0_4', onyx.home + '/data/htk_r1_adg0_4')
    train_model(in_filename, mlf_filename, scp_filename, out_filename, scp_prefix_subst=prefix_subst)


def usage(prog_name):
    print("Usage: %s [infile mlf_file scp_file] [outfile]" % (prog_name,))
    print("With no arguments, run doctests; with outfile, run test0 and write output to outfile")
    print("With all arguments, load infile, train with mlf_file and scp_file, and write output to outfile")
    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        from onyx import onyx_mainstartup
        onyx_mainstartup()
    elif len(sys.argv) == 2:
        test0(sys.argv[1])
    elif len(sys.argv) == 5:
        train_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        usage(sys.argv[0])


