import os
import sys
import pandas as pd
import json
import numpy as np
import csv
from scipy.special import comb


def myround(x, base):
    return base * round(x/base+(x%base>0))

CONFIG_PATH='config.csv'

def generate_config():    
    L=5
    op_weight_vec=np.arange(1,L+1,1)
    Hind_vec=np.arange(0,1) ## average over Hamiltonian initiations
    #niter=2 ## average over operators (how  many do we sample per weight?) 
            ## could also make this weight-dependent and define niter in opweight loop
    #chunk_size=1 #break them into chunks
    
    cput="00:10:00"
    mem="12"
    
    params=[]
    header = ["L", "op_weight","Hind","niter", "chunk_size", "cput", "mem"]
    with open(CONFIG_PATH, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(header)
        for op_weight in op_weight_vec:
            for Hind in Hind_vec:
                if op_weight>2:
                    fraction_of_all_ops=myround(int(comb(L,op_weight)*3**op_weight)//50,100)
                    niter=fraction_of_all_ops
                    chunk_size=fraction_of_all_ops//10
                else:
                    niter=3*L
                    chunk_size=niter
                print(['L=', L, ', op_weight=', op_weight, ', Hind=',Hind,', niter=', niter, 'chunk_size=', chunk_size])
                writer.writerow([L,op_weight,Hind, niter, chunk_size, cput, mem])

directory = sys.argv[1]
CONFIG_PATH=os.path.join(directory, CONFIG_PATH)

if not os.path.exists(directory):
    try:
        os.mkdir(directory)
    except OSError:
        print ("Creation of the directory %s failed" % directory)

generate_config()
config = pd.read_csv(CONFIG_PATH)
for index, row in config.iterrows():
    print(index)
    params = {col:row[col] for col in config.columns}
    params['directory'] = directory
    cput = params["cput"]
    mem = params["mem"]
    niter = params["niter"]
    chunk_size = params["chunk_size"]
    replace_tokens = {b'__CPUT__':str.encode(str(cput)), b'__MEM__':str.encode(str(mem)), b'__NAME__':str.encode('L'+str(params['L']))}
    with open('cheryne.qusub.template', 'rb') as fi, open(os.path.join(directory, 'cheryne.qusub'), 'wb') as fo:
        for line in fi:
            for token in replace_tokens:
                line = line.replace(token, replace_tokens[token])
            fo.write(line)
    num_jobs = niter // chunk_size 
    for i in range(num_jobs):
        params['chunk_id'] = i
        params_string = json.dumps(params, separators=(',',':'))
        command = 'bash {}/cheryne.qusub {} {} \'{}\''.format(directory, cput, mem, params_string)
        os.system(command)

    if niter % chunk_size != 0:
        params['chunk_id'] = num_jobs
        params['chunk_size'] = niter % chunk_size
        params_string = json.dumps(params, separators=(',',':'))
        command = 'bash {}/cheryne.qusub {} {} \'{}\''.format(directory, cput, mem, params_string)
        os.system(command)


