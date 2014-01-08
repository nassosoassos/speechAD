'''
Created on Dec 29, 2011

@author: nassos
'''
import os
from my_utils import which
import subprocess

os.environ['PATH'] += os.pathsep + '/usr/local/bin'
hparse_bin = which.which('HParse')

def create_trivial_grammar(labels, grammar_file, mode='sequence'):
    gr_file = open(grammar_file,'w')
    gr_file.write('$class = ')
    gr_file.write(labels[0])
    for l in labels[1:]:
        gr_file.write(' | '+l)
    gr_file.write(';\n')
    if mode=='sequence':
        gr_file.write('(<$class>)')
    elif mode=='single':
        gr_file.write('($class)')
    gr_file.close() 

def create_trivial_wordnet(labels,wdnet_file, mode='sequence'):
    grammar_file = wdnet_file+'.gram'
    create_trivial_grammar(labels,grammar_file, mode)
    cmd = [hparse_bin,grammar_file,wdnet_file]
    print(os.getcwd())
    print(cmd)
    subprocess.call(cmd) 
    

if __name__ == '__main__':
    pass
