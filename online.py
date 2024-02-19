import numpy as np
import ast
from utils.utils import *
from tqdm import tqdm
import pandas as pd
import scipy.stats as stats
import faiss
from utils.utils import *

def make_initial_cache(df, NUM, embed):
    cache = []
    for idx in tqdm(range(len(df))):
        if idx < NUM: 
            aux = ast.literal_eval(df.iloc[idx][embed])
            if len(cache)==0: cache = np.array(aux)
            else: cache = np.vstack((cache, aux))
    return cache


contador = -1

method2embed = {'CS1': 'mpnet', 'CS2': 'simcse'}
df2 = pd.DataFrame(columns=['isear', 'rt-polarity', 'fever', 'openbook'])
for TASK in ['isear', 'rt-polarity', 'fever', 'openbook']:
    for N_INIT in [500, 1000, 2000, 3000]:
        contador += 1
        BT, EN, MV, MV_dis,  MA, acc, acc_llm, tgt_llm, BT_llm, results = load(TASK, 0, str(N_INIT), LIMIT=10000) #AQUEST LIMIT SHA DE REVISAR
        df = pd.read_csv('../data/'+TASK+'/train_soft_embeddings.csv')
        df = df.iloc[df.prova]
        all = []
        threshold=-0.5
        all_rdm = []
        all_bo = []
        BUDGETS=np.linspace(2, len(BT), 20) # ha DE SER 20
        #BUDGETS=BUDGETS[:-1]

        plotato = {}
        METHODS = ['RD', 'BT', 'EN', 'CS1', 'CS2']
        if int(N_INIT)>=500: METHODS.append('MV')
        no_rdm = [METHOD for METHOD in METHODS if METHOD != 'RD']   
        for METHOD in METHODS: 
            plotato[METHOD] = []
        for BUDGET in BUDGETS:
            cache = {}
            res = faiss.StandardGpuResources()
            gpu_index_flat = {}
            for METHOD in ['CS1', 'CS2']:
                cache[METHOD] = make_initial_cache(df, N_INIT, method2embed[METHOD])
                d = 768
                res = faiss.StandardGpuResources()
                index_flat = faiss.IndexFlatL2(d)
                gpu_index_flat[METHOD] = faiss.index_cpu_to_gpu(res, 0, index_flat)
                gpu_index_flat[METHOD].add(cache[METHOD])   
            BUDGET=int(BUDGET) 
            budget_methods = {} 
            acc_methods = {}
            for METHOD in METHODS: 
                budget_methods[METHOD] = BUDGET
                acc_methods[METHOD] = 0
            history = {}
            history['CS1'] = []
            history['CS2'] = []
            for idx in range(len(EN)):
                src = {}
                for METHOD in ['CS1', 'CS2']:
                    src[METHOD] = ast.literal_eval(df.iloc[N_INIT+idx][method2embed[METHOD]])
                    src[METHOD] = np.array(src[METHOD]).reshape((1,768))
                    D, I = gpu_index_flat[METHOD].search(src[METHOD], 1)
                    aux_dis = 1-D[0]
                    history[METHOD].append(aux_dis)
                history['BT'] = BT[:idx+1]
                history['EN'] = EN[:idx+1]
                history['MV'] = MV_dis[:idx+1]
                llm_output = acc_llm[idx]
                student_output = acc[idx]
                for METHOD in no_rdm:
                    b = budget_methods[METHOD]
                    if b>0:
                        mean = np.mean(history[METHOD])
                        std = np.std(history[METHOD])
                        if idx == 0: 
                            TGT_PERCENT = 100
                            percent = 10
                        else:
                            current = history[METHOD][-1]
                            TGT_PERCENT = b/(len(BT)-idx)
                            z = (current-mean)/std
                            percent = stats.norm.cdf(z)
                            
                        if percent <= TGT_PERCENT: 
                            budget_methods[METHOD] -= 1
                            acc_methods[METHOD] += llm_output
                            if METHOD in ['CS1', 'CS2']:
                                cache[METHOD] = gpu_index_flat[METHOD].add(src[METHOD])   
                        else:
                            acc_methods[METHOD] += student_output
                    else: 
                        acc_methods[METHOD] += student_output
                if budget_methods['RD'] > 0:
                    budget_methods['RD'] -= 1
                    acc_methods['RD'] += llm_output
                else:
                    acc_methods['RD'] += student_output
            for METHOD in METHODS:
                acc_methods[METHOD] /= len(BT) #BUDGET #he canviat aixo
                #aquest assert falla!
                #assert budget_methods[METHOD] == 0 or budget_methods[METHOD] == 1
                plotato[METHOD].append(acc_methods[METHOD])

        tran_name={'CS1':'Coreset (mpnet)', 'CS2':'Coreset (simcse)', 'EN': 'Entropy', 'BT':'Margin Sampling', 'MV':'Query by Committee', 'b1':'Front-loading', 'RD':'Random', 'BT_non_pred':'BT_non_pred'}
        style={'Margin Sampling':'b-', 'Coreset (mpnet)':'g-', 'Entropy':'y-', 'Coreset (simcse)':'c-', 'Query by Committee':'r-', 'Random':'k-', 'BT_non_pred':'m-'}
        target = ['Coreset (mpnet)', 'Coreset (simcse)', 'Entropy', 'Margin Sampling', 'Query by Committee', 'Random', 'BT_non_pred']
        plt.figure(contador)
        for METHOD in METHODS:
            plt.plot(BUDGETS, plotato[METHOD], style[tran_name[METHOD]], label=tran_name[METHOD])
            df2.at[(METHOD+'_'+str(N_INIT), TASK)] = auc(BUDGETS, plotato[METHOD])/BUDGETS[-1]

        order = []
        handles, labels = plt.gca().get_legend_handles_labels()
        for element in target:
            if element in labels:
                order.append(labels.index(element))

        plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right')
        plt.xlabel('Budget (number of LLM calls)')
        plt.ylabel('Accuracy online')
        plt.title(TASK.upper())
        plt.savefig('plots/online_new3/'+TASK+'_'+str(N_INIT)+'.png')
print(df2)
df2.to_csv('plots/online_new3/online_new.csv')