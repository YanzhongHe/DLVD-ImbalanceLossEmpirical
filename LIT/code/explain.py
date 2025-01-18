import pandas as pd
import pickle

from linevul_explainer import LineVulModelExplainer, Args
from lit_nlp.components import gradient_maps
import sys
import json
import pickle
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


# def get_data(file_path='../data/big-vul_dataset/test.csv', top_x=3):
#   df = pd.read_csv(file_path)
#   funcs = df["func"].tolist()
#   labels = df["target"].tolist()
#   input = []
#   for i in range(len(funcs)):
#     input.append({
#         'sentence': funcs[i],
#         'label': labels[i]
#     })
#   return input


def get_data(file_path='/root/autodl-tmp/LineVul/data/BIGVUL4LIT/json/LIT-Bigvul_1.json', top_x=3):
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


inputs = get_data(file_path='/root/autodl-tmp/LineVul/data/BIGVUL4LIT/json/LIT-Bigvul_1.json')

print("Model loading")
args = Args()
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
# pickle.dump(ig_results,open('explanation_LineVul_LIT_LALoss_ReVeal-D_test.pkl', 'wb'))
pickle.dump(ig_results,open('/root/autodl-tmp/LineVul/LIT/code/explanation_CBCELoss-Bigvul-1.pkl', 'wb'))
print("Explanation Done for {}".format(len(ig_results)))




