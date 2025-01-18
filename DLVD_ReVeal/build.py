import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--label_rate", required=True, help="Choose the proportion of labels. ")  
parser.add_argument("--dataset", required=True, help="Select dataset")  
parser.add_argument("--stopper_mode", required=True, help="Select model stop type")  
parser.add_argument("--base_model", required=True, help="base model")  
parser.add_argument("--sampling", required=True, help="Sampling method")  
parser.add_argument("--loss_fct", required=True,
                    help="Loss function type")  
parser.add_argument("--comment", required=True,
                    help="model describe")  

args = vars(parser.parse_args())
if args['base_model'] == 'ReVeal':
    from ReVeal.train import *

os.environ['TOKENIZERS_PARALLELISM'] = 'False'


def build_teacher(seeds, loss_fct1):
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    with open(f'{args["base_model"]}/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    setup_seed(config['train']['random_seed'])

    args["loss_fct"] = loss_fct1
    args["comment"] = loss_fct1

    config['data']['label_rate'] = args['label_rate']
    config['data']['ckpt_path'] = f'{args["base_model"]}/best_model'
    config['train']['stopper_mode'] = args['stopper_mode']
    config['loss']['loss_type'] = args["loss_fct"]
    config['loss']['comment'] = args["comment"]
    ckpt_path = config['data']['ckpt_path'] + '/' + args["dataset"] + '/' + args["comment"] + '/seed' + str(
        seeds) + '/' + args["label_rate"] + '/' + config['train']['stopper_mode']
    config['data']['ssl_data_path'] = f'{args["base_model"]}/VD_data/{args["dataset"]}'

    print("#######")
    print(ckpt_path)
    print(config['loss']['loss_type'])
    print(config['data']['ssl_data_path'])
    print("-----------")
    print(config['loss']['loss_type'])
    print(config['loss']['comment'])
    print("#######")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    with open(f'{ckpt_path}/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data=config, stream=f)

    train_teacher(ckpt_path, config, seed_s=seeds)


Seeds = [1, 2, 3, 4, 5]

loss_fcts = ["celoss","focalloss","CBfocalloss","CBceloss","GHMloss","LDAMloss","LAloss","WCEloss","DiceLoss","DSCLoss"]

for loss_fct_ in loss_fcts:
    for seed_ in Seeds:
        if __name__ == '__main__':
            build_teacher(seeds=seed_, loss_fct1=loss_fct_)
