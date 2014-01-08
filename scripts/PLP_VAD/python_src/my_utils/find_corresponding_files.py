'''
Created: January 27th 2012

Description: 
  Given a list of files for each file it tries to find a single file with the same basename and 
  a given suffix in the given directories. It returs both the original and the found files in 
  an scp file with the given name.
'''
import argparse
import fnmatch
import logging
import os

from my_utils import lists

def find_corresponding_files(in_files, search_directories, suffix):
    '''
    Given a list of files, find the corresponding files with the same basename
    and probably a differnt suffix in the given search directories
    '''
    found_files = []
    for fl in in_files:
        b_name = os.path.splitext(os.path.split(fl)[1])[0]
        f_name = '{}.{}'.format(b_name, suffix)
        matches = []
        for dr in search_directories:
            matches.extend(find_file_in_dir_recursive(f_name, dr))

        len_matches = len(matches)
        if len_matches > 1:
            logging.warning('In find_corresponding_files more than one matches found {}'.format(f_name))
        if len_matches == 0:
            matches.append(' ')
        found_files.append(matches[0])


    return(found_files)

def find_file_in_dir_recursive(fname, folder):
    '''
    Find all the files following a certain name pattern 
    recursively in a folder
    '''
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, fname):
                matches.append(os.path.join(folder, root, filename))
    return(matches)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument('--in_list', help='input file list', required=True)
    parser.add_argument('--search_dir', action='append', help='directory to search in')
    parser.add_argument('--suffix', default='txt', help='suffix of the files to be found')
    parser.add_argument('--scp_file', help='scp file where all the information will be stored', required=True)
    parser.add_argument('--reverse', help='write the found files first in the scp file', action='store_true')

    args = parser.parse_args()

    in_file_list = args.in_list
    search_directories = args.search_dir
    scp_file = args.scp_file
    sfx = args.suffix

    in_files = lists.get_contents(in_file_list)
    corresponding_files = find_corresponding_files(in_files, search_directories, sfx)

    scp = open(scp_file,'w')
    for i_fl, c_fl in zip(in_files, corresponding_files):
        if args.reverse:
            scp.write('{} {}\n'.format(c_fl, i_fl))
        else:
            scp.write('{} {}\n'.format(i_fl, c_fl))


    scp.close()

