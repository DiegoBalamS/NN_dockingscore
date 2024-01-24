import numpy as np
import pandas as pd

def vocab(smile_file):
    df=pd.read_table(smile_file)
    smiles=df['smiles'].values
    pro_sms=[]
    for sm in smiles:
        sm = ' '.join(list(sm))
        before = before = ['C l -', 'C l', 'O -', 'N +', 'n +', 'B r -', 'B r', 'N a +', 'N a', 'I -', 'S i']
        after = ['Cl-', 'Cl', 'O-', 'N+', 'n+', 'Br-', 'Br', 'Na+', 'Na', 'I-', 'Si']
        for b,a in zip(before, after):
            sm = sm.replace(b, a)
        pro_sms.append(sm)

    df['processed_smiles']=pro_sms
    
    vocab=[]
    for sm in smiles:
        l=sm.split(' ')
        for w in l:
            if w not in vocab:
                vocab.append(w)

    with open('project/vocab.txt','a') as f:
        for v in vocab:
            f.write(v+'\n')

