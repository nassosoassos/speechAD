'''
Created on Jan 15, 2012

@author: nassos
'''
import logging
import os

def create_corresponding_list_abs_paths(in_file_list,in_path,out_file_list):
    '''
    Create a corresponding list of files in a different directory
    '''
    try:
        in_list = open(in_file_list,'r')
    except:
        logging.exception("Error opening file {} for reading".format(in_file_list))
        raise
    try:
        out_list = open(out_file_list,'w')
    except:
        logging.exception("Error opening file {} for writing".format(out_file_list))
        raise

    try:
        for fl in in_list:
            out_list.write(os.path.join(in_path,fl))
    except IOError:
        logging.exception('Problem writing, reading file.')
        raise
    finally:
        in_list.close()
        out_list.close()



def create_corresponding_list_assert(in_file_list, out_dir, out_file_list, out_sfx):
    '''
    Change the extension of a filename, check if the new file exists
    and if yes write it in the out_file_list.
    '''
    try:
        in_list = open(in_file_list,'r')
        out_list = open(out_file_list,'w')

        for fl in in_list:
            fname = os.path.splitext(os.path.split(fl.rstrip('\r\n'))[1])[0]
            out_file = os.path.join(out_dir,fname+'.'+out_sfx)
            if os.path.exists(out_file):
                out_list.write('{0}\n'.format(out_file))
    except IOError:
        logging.exception('Problem writing, reading file.')
    finally:
        in_list.close()
        out_list.close()

def create_lists_from_scp(scp=None, list_of_lists=[]):
    '''
    Read a multicolumn space separated scp file, separate the columns and write them
    into different files
    '''
    scp_file = open(scp,'r')

    f_handles = []
    for fl in list_of_lists:
        f_handles.append(open(fl,'w'))
    n_files = len(f_handles)

    for ln in scp_file:
        ln = ln.rstrip('\r\n')
        elms = ln.split()
        n_elms = len(elms)
        for count in range(n_elms):
            if count<n_files:
                f_handles[count].write('{0}\n'.format(elms[count]))

    for fh in f_handles:
        fh.close()


def get_contents(in_list):
    '''
    Get the filelist contents in a list
    '''
    i_l = open(in_list,'r')
    contents = []
    for ln in i_l:
        contents.append(ln.rstrip('\r\n'))
    i_l.close()
    return(contents)

if __name__ == '__main__':
    pass
