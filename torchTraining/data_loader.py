# https://discuss.pytorch.org/t/loading-huge-data-functionality/346

from torchnet.dataset import ListDataset
import torch
import string
import torch.utils.data as data_utils

# "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + " .,;'"
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
def encoding(string : str):
    '''
    character 단위로 숫자를 assign하고 list로 반환
    '''
    char_list = []
    for char in string:
        char_list.append(all_letters.find(char))
    return char_list


def load_func(line):
    # a line in 'list.txt"
    src, target = line.split('\t')
    src = encoding(src)
    target = encoding(target)

    return {'src': src, 'target': target}

def batchify(batch):
    '''
    batch = [{'src' : 위에 정의한거, 'dst' : 위에 정의한거}]
    '''
    def getMaxLen(batch, target):
        '''
        batch에서 가장 긴 녀석을 가져온다.
        '''
        max_len = 0
        for item in batch:
            if len(item[target]) > max_len:
                max_len = len(item[target])
        return max_len
    src_len = getMaxLen(batch, 'src')
    target_len = getMaxLen(batch, 'target')

    batch_src = torch.zeros(src_len, len(batch), n_letters)
    batch_target = torch.zeros(target_len, len(batch), n_letters)

    for idx, item in enumerate(batch):
        for j in range(len(item['src'])):
            batch_src[j][idx][item['src'][j]] += 1
        for j in range(len(item['target'])):
            batch_target[j][idx][item['target'][j]] += 1

    return {'src': batch_src, 'target': batch_target} # you can return a tuple or whatever you want it to


dataset = ListDataset('test', load_func) #list.txt contain list of datafiles, one per line
dataset = data_utils.DataLoader(dataset=dataset, batch_size=50, num_workers=8, collate_fn=batchify) #This will load data when needed, in parallel, up to <num_workers> thread.

for x in dataset: #iterate dataset
    print(x)