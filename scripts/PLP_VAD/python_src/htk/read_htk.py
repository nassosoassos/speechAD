'''
Created on Dec 28, 2011

@author: nassos
'''
import argparse
import struct
import textwrap
import os

def read_htk_header(htk_file):
    assert(os.path.exists(htk_file))
    htk_file_id = open(htk_file,"rb")
    n_samples = 0
    sampPeriod = 0
    sampSize = 0
    parmKind = 0
    try:
        n_samples = struct.unpack('>I',htk_file_id.read(4))[0]
        sampPeriod = struct.unpack('>I',htk_file_id.read(4))[0]
        sampSize = struct.unpack('>H',htk_file_id.read(2))[0]
        parmKind = struct.unpack('>H',htk_file_id.read(2))[0]
    finally:
        htk_file_id.close()
    
    return (n_samples, sampPeriod, sampSize, parmKind)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read HTK formatted file. Currently only showing the header',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
                                     Example usage: 
                                         read_htk.py file.mfc                                     
                                     '''))
    parser.add_argument('htk_file', metavar='htk_file',
                        type=str, help='HTK-formatted file' )
    args = parser.parse_args()

    (n_samples, sampPeriod, sampSize, parmKind) = read_htk_header(args.htk_file)
    print('N_samples: {}, sampPeriod: {}, sampSize: {}, parmKind: {}'.format(n_samples, sampPeriod,sampSize,parmKind))
