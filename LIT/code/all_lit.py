import pandas as pd
import pickle

from linevul_explainer import LineVulModelExplainer, Args
from lit_nlp.components import gradient_maps
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import json



def get_data(file_path='code.json', top_x=3):
    df = pd.read_json(file_path)
    funcs = df["processed_func"].tolist()
    labels = df["target"].tolist()
    input = []
    for i in range(len(funcs)):
        input.append({
            'sentence': funcs[i],
            'label': labels[i]
        })
    return input

def get_max_match(original, start_index,  tokens):
    lo = len(original)
    lt = len(tokens)
    ind , max_score = 0, lo
    for i in range(start_index, lo-lt+ 1):
        found = True
        for j in range(lt):
            if  original[i + j] != tokens[j]:
                found = False
        if found:
            return [x for x in range(i, i + lt)]
    return []


def calculate_score(scores, ids):
    sum = 0
    for id in ids:
        sum += scores[id]
    return sum





inputs = get_data(file_path='../code/example/big-vul/8040/data_8040.json')

print("Model loading")
args = Args(output_dir="../code/example/big-vul/CBFocalLos_LineVul_seed1/linevul_cbfocalloss")
model = LineVulModelExplainer(args)
model.load_model()
# model.activate_evaluation()


print(model.output_spec())


ig = gradient_maps.IntegratedGradients()

print("Input Data size: ", len(inputs))
ig_results = []
for i in range(0, len(inputs), 128):
  ig_results += ig.run(inputs[i : i + 128], model, inputs)
  print("Finished from {} to {}, result length {}".format(i, i + 128, len(ig_results)))
print("Finished ", len(ig_results))

pickle.dump(ig_results,open('../code/example/big-vul/8040/explanation_seed1_linevul_CBFocalLoss_Bigvul.pkl', 'wb'))
print("Explanation Done for {}".format(len(ig_results)))



with open("../code/example/big-vul/8040/data_8040.json", 'r') as f:
    data = []
    for  line in f:
        try:
            print(line.strip())
            data.append(json.loads(line.strip()))
        except Exception as e:
            print(e)

config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
tokenizer = tokenizer_class.from_pretrained('../../DLVD_LineVul/bert')
explanations = pickle.load(open('../code/example/big-vul/8040/explanation_seed1_linevul_CBFocalLoss_Bigvul.pkl', 'rb'))


input_file = '../code/example/big-vul/8040/data_8040.json'
with open(input_file, 'r') as file:
    data = json.load(file)

results = []

for idx, func in data['processed_func'].items():

    target = data['target'][idx]
    sample = {
        'idx': idx,
        'func': func,
        'target': target
    }


    explanation = explanations[0]


    original_tokens = ['<s>'] + tokenizer.tokenize(sample['func'])[:4096] + ['</s>']
    explanation_tokens = explanation['token_grad_sentence'].tokens
    explanation_scores = explanation['token_grad_sentence'].salience

    if original_tokens != explanation_tokens:
        print(sample['idx'])
    
    lines = sample['func'].split("\n")
    start_index = 0
    lines_scores = []
    cnt = 0

    for line in lines:
        line_tokens = tokenizer.tokenize(line)
        ids = get_max_match(explanation_tokens, start_index, line_tokens)
        if ids:
            start_index = ids[-1] + 1
            cnt += 1
        lines_scores.append(calculate_score(explanation_scores, ids))

    assert cnt > 0
    
    # 将处理结果存储在 results 中
    results.append({
        'idx': sample['idx'],
        'scores': lines_scores,
    })

print(results)

json.dump(results, open(f"../code/example/big-vul/8040/line_scores-seed1_linevul_CBFocalLoss_Bigvul.json", 'w'))
