###########################################################################
#
# File:         rocutils.py (directory: ./py/onyx/util)
# Date:         Mon 10 Mar 2008 18:34
# Author:       Ken Basye
# Description:  Utility code for generating ROC and DET curves
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008, 2009 The Johns Hopkins University
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
    Utilities for generating ROC and DET curves 
"""
import StringIO

def _uniquify_preserving_first(iterable, eq_pred):
    item = iterable.next()
    while 1:
        try:
            next_item = iterable.next()
        except:
            yield item
            break
        if not eq_pred(item, next_item):
            yield item
            item = next_item
            
def _uniquify_preserving_last(iterable, eq_pred):
    item = iterable.next()
    while 1:
        try:
            next_item = iterable.next()
        except:
            yield item
            break
        if not eq_pred(item, next_item):
            yield item
            item = next_item
        else:
            item = next_item
                        
def make_ROC_data(reference, ratios):
    """
    reference is a list of 0/1 values which are the correct classifications
    values is a parallel list of numeric values, with higher values intending to
    map toward classifications of 1.
    
    Returns data for a ROC curve in the form of a list of triples, where each triple
    contains an interesting threshold value, the fraction of correct identifications (true positives)
    as a percent, and the fraction of false positives, at that threshold.  The triples are
    ordered by threshold from lowest (fewest false positives) to highest (most true positives)

    Note that a typical ROC curve would plot false_pos on the X axis and true_pos on the Y axis
    using a linear scale.
    
    >>> ref = [0,0,0,0,0,1,1,1,1,1]
    >>> values = [2, 3, 4, 9, 4, 5, 6, 9, 9, 3]
    >>> res = make_ROC_data(ref, values)
    >>> res
    [(0.0, 0.0, 9), (20.0, 80.0, 4), (80.0, 100.0, 2)]
    """
    det_data = make_DET_data(reference, ratios)
    roc_data = [(fp, 100-miss, t) for (fp, miss, t) in det_data]
    return roc_data


def make_DET_data(reference, ratios):
    """
    reference is a list of 0/1 values which are the correct
    classifications values is a parallel list of numeric values, with
    higher values intending to map toward classifications of 1.
    
    Returns data for a DET curve in the form of a list of triples,
    where each triple contains the fraction of false positives as a
    percent, the fraction of false negatives, and the threshold value
    that generated those rates.  The triples are ordered by threshold
    from lowest (fewest false positives) to highest (fewest misses)

    Note that a typical DET curve would plot false_pos on the X axis
    and false_neg on the Y axis, oftentimes with a normal deviate
    scale.

    >>> ref = [0,0,0,0,0,1,1,1,1,1]
    >>> values = [2, 3, 4, 9, 4, 5, 6, 9, 9, 3]
    >>> res = make_DET_data(ref, values)
    >>> res
    [(0.0, 100.0, 9), (20.0, 19.999999999999996, 4), (80.0, 0.0, 2)]

    """

    assert( len(reference) == len(ratios) )
    num_pos = reference.count(1)
    num_neg = reference.count(0)
    assert( num_pos + num_neg == len(reference))

    full_result = []

    # Find the list of interesting threshholds, which is any value in
    # the list of ratios

    # Seems like there should be an easier way to uniquify a list
    all_threshes = set(ratios)
    all_threshes = list(all_threshes)
    all_threshes.sort()

    def count_values_over_thresh(value, ref, ratios, t):
        result = 0
        for (i, r) in enumerate(ratios):
            if ref[i] == value and r > t:
                result += 1
        return result

    # Now find precision and recall at each threshold
    for thresh in all_threshes:
        num_neg_accepted = count_values_over_thresh(0, reference, ratios, thresh)
        num_pos_accepted = count_values_over_thresh(1, reference, ratios, thresh)
        full_result.append((100 * float(num_neg_accepted) / num_neg,  # false positives
                            100 * (1 - float(num_pos_accepted) / num_pos),  # misses
                           thresh))

    def eq0(x,y): return x[0] == y[0]
    def eq1(x,y): return x[1] == y[1]
        
    iter1 = _uniquify_preserving_first(iter(full_result), eq0)
    ret = list(_uniquify_preserving_last(iter1, eq1))
    ret.reverse()
    return ret


def write_data_as_csv(data, stream, header_type = "DET"):
    """ Write either ROC or DET data as comma-separated text, suitable for import into
    a spreadsheet or other tool.  Writes DET header fields be default, use header_type
    of "ROC" or None for ROC headers or no headers, respectively.

    >>> ref = [0,0,0,0,0,1,1,1,1,1]
    >>> values = [2, 3, 4, 9, 4, 5, 6, 9, 9, 3]
    >>> res = make_DET_data(ref, values)
    >>> s = StringIO.StringIO()
    >>> write_data_as_csv(res, s)
    >>> out = s.getvalue()
    >>> print out
    False Alarm Rate,  Miss Rate,  Threshold
    0.0,  100.0,  9
    20.0,  20.0,  4
    80.0,  0.0,  2
    <BLANKLINE>
    >>> s.seek(0)
    >>> res = make_ROC_data(ref, values)
    >>> write_data_as_csv(res, s, header_type="ROC")
    >>> out = s.getvalue()
    >>> print out
    False Pos Rate, True Pos Rate, Threshold
    0.0,  0.0,  9
    20.0,  80.0,  4
    80.0,  100.0,  2
    <BLANKLINE>
    >>> s.close()
    """
    if header_type == "DET":
        stream.write("False Alarm Rate,  Miss Rate,  Threshold")
    elif header_type == "ROC":
        stream.write("False Pos Rate, True Pos Rate, Threshold")
    [stream.write("\n%s,  %s,  %s" % triple) for triple in data]
    stream.write("\n")    

def _test0():
    ref = [0,0,0,0,0,1,1,1,1,1]
    values = [2, 3, 4, 9, 4, 5, 6, 9, 9, 3]
    res = make_DET_data(ref, values)
    s = open("foo_csv.txt", "w")
    write_data_as_csv(res, s)
    s.close()
    

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
#   _test0()
