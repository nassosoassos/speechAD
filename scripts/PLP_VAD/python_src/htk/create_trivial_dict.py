'''
Created on Dec 29, 2011

@author: nassos
'''

def create_trivial_dict(labels,dict_file):
    d_file = open(dict_file,'w')
    for l in labels:
        d_file.write('{0} {1}\n'.format(l,l))
    d_file.close()

if __name__ == '__main__':
    pass
