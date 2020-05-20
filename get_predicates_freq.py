import dill
import numpy as np
f=open('predcls_sg_constrain', 'rb')
cache=dill.load(f)
train_keys=[]
for key in cache.keys():
    train_keys.append(key)

f.close()
f2=open('processed_triplets','rb')
cache=dill.load(f2)
pred_freq=np.zeros((51,))
for key in train_keys:
    selected_label = cache[key]['selected_label']
    if np.size(selected_label) ==0:
        continue
    pred_freq[selected_label[0][2]] += 1

np.save('no_sg_selected_predicate_freq', pred_freq)