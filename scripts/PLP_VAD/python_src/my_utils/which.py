'''
Created on Dec 27, 2011

@author: nassos
'''
import argparse

def which(program):
    import os
    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    fpath = os.path.split(program)[0]
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python variant of the unix command which')
    parser.add_argument('program', type=str, help='the command whose path is requested' )
         
    args = parser.parse_args()
    command = which(args.program)
    if command != None:
        print(command)