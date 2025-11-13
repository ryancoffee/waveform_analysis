#!/usr/bin/python3

import utils 
import h5py
import os

def collect_files():
    collection = {}
    roots = []
    filenames = [f for f in os.listdir(os.getenv('_datapath')) if os.path.isfile(os.path.join(os.getenv('_datapath'),f))]
    ext = os.path.splitext(filenames[0])[-1]
    if len(filenames) ==0:
        print('No files found')
    else:
        key = os.getenv('_datapath').split(os.sep)[-1]
        collection.update({key:{}})
        for i,fname in enumerate(filenames):
            if i%100 == 0:
                print('working %s'%fname)
            roots += [ os.path.splitext(fname)[0] ]
            if 'data' not in collection[key].keys():
                    collection[key].update( { 'data':[] } )
            if ext == '.txt':
                collection[key]['data'] += [utils.txt2array( os.path.join(os.getenv('_datapath'),fname ) ) ]    
            elif ext == '.trc':
                collection[key]['data'] += [utils.trace2array( os.path.join(os.getenv('_datapath'),fname) ) ] 
            else:
                collection[key].update({'data': []  } )
                print('unknown file extension')
        collection[key].update({'ext': ext})
        collection[key].update({'roots': roots})
    return collection


def writeH5(d):
    with h5py.File(os.path.join(os.getenv('_resultpath'),'results_collection.h5'),'a') as f:
        for k in d.keys():
            if k in f.keys():
                del f[k]
            grp = f.create_group(k)
            for l in d[k].keys():
                if l == 'data':
                    grp.create_dataset('data',data=d[k][l])
                else:
                    grp.attrs.create(l,data=d[k][l])


def main():
    writeH5( collect_files() )

if __name__ == '__main__':
    if os.path.isdir(os.getenv('_datapath')) and os.path.isdir(os.getenv('_resultpath')):
        main()
    else:
        print('First you must set_vars.bash <infilepath> <outfilename>')

