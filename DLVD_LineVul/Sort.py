parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str)
parser.add_argument('--savepath', type=str)
opts = parser.parse_args()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def sort_base_padding(data,labels):

    def key_sort_function(x):
        x = data[x]
        token_length = len(tokenizer(x)['input_ids'])
        return token_length

    idx_list = list(labels.keys())

    # aim_list:[(data,label:str),]
    idx_list.sort(key=key_sort_function)

    return idx_list
