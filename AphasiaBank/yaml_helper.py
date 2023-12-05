'''
Yaml helper
'''

import argparse
import yaml
import os
import shutil
from hyperpyyaml import load_hyperpyyaml,dump_hyperpyyaml
from io import StringIO 
import torch


def create_yaml_torre(args):
    # Create new yaml and return path to generated yaml
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base_torre.yaml"
    # base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(exp_name, sev_arg)
    
    # outfile = f"/y/mkperez/speechbrain/AphasiaBank/hparams/{exp_name}/{sev_arg}.yaml"
    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    shutil.copyfile(base_yaml, outfile)


    
    if args.severity == "freeze_w2v":
        freeze = "True"
        grad_acc = "4"
        batch_size = "24"
        epochs = "20"
        lr_w2v2 = "0.0001"
        lr_anneal_factor_w2v2 = "1.0" # og = 0.9
        lr_decoder = "0.9"
        lr_anneal_factor_decoder = "0.8" # og = 0.8
        patience = "1"
        data_dir = "/y/mkperez/speechbrain/AphasiaBank/data/personalization_torre"
        mtl = "False"
        train_data = "train_no_kansas.csv" # [train_no_kansas.csv, train_all.csv]
        val_data = "val_no_kansas.csv"
        test_data = "test_kansas.csv"
        tr_str = train_data.split(".")[0]
        val_str = val_data.split(".")[0]
        exp_out_folder = f"results/{exp_name}/freeze-{freeze}/tr-{tr_str}_val-{val_str}"

    elif args.severity == "last2_w2v":
        freeze = "False"
        grad_acc = "16"
        batch_size = "3"
        epochs = "30"
        lr_w2v2 = "0.001"
        lr_anneal_factor_w2v2 = "0.9" # og = 0.9
        lr_decoder = "0.9"
        lr_anneal_factor_decoder = "0.8" # og = 0.8
        patience = "1"
        mtl = "False"
        data_dir = "/y/mkperez/speechbrain/AphasiaBank/data/personalization_torre"
        train_data = "train_all.csv" # [train_no_kansas.csv, train_all.csv]
        val_data = "val_no_kansas.csv" # [val_kansas.csv, val_no_kansas.csv]
        test_data = "test_kansas.csv"
        tr_str = train_data.split(".")[0]
        val_str = val_data.split(".")[0]
        exp_out_folder = f"results/{exp_name}/freeze-{freeze}/tr-{tr_str}_val-{val_str}"


    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('freeze_PLACEHOLDER', f"{freeze}")
        filedata = filedata.replace('grad_acc_PLACEHOLDER', f"{grad_acc}")
        filedata = filedata.replace('batch_size_PLACEHOLDER', f"{batch_size}")
        filedata = filedata.replace('epoch_PLACEHOLDER', f"{epochs}")
        filedata = filedata.replace('lr_w2v2_PLACEHOLDER', f"{lr_w2v2}")
        filedata = filedata.replace('lr_decoder_PLACEHOLDER', f"{lr_decoder}")
        filedata = filedata.replace('mtl_PLACEHOLDER', f"{mtl}")
        filedata = filedata.replace('train_data_PLACEHOLDER', f"{train_data}")
        filedata = filedata.replace('val_data_PLACEHOLDER', f"{val_data}")
        filedata = filedata.replace('test_data_PLACEHOLDER', f"{test_data}")
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_dir}")
        filedata = filedata.replace('exp_out_folder_PLACEHOLDER', f"{exp_out_folder}")
        filedata = filedata.replace('lr_anneal_factor_w2v2_PLACEHOLDER', f"{lr_anneal_factor_w2v2}")
        filedata = filedata.replace('lr_anneal_factor_decoder_PLACEHOLDER', f"{lr_anneal_factor_decoder}")
        filedata = filedata.replace('patience_PLACEHOLDER', f"{patience}")
        

        with open(outfile,'w') as fout:
            fout.write(filedata)

def create_yaml_personalization(args):
    # Create new yaml and return path to generated yaml
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base_personalization.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(exp_name, sev_arg)
    
    # outfile = f"/y/mkperez/speechbrain/AphasiaBank/hparams/{exp_name}/{sev_arg}.yaml"
    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    shutil.copyfile(base_yaml, outfile)


    if args.severity == "personalize":
        freeze = "False"
        grad_acc = "1"
        batch_size = "3"
        epochs = "50"
        lr_w2v2 = "0.00008" # 0.00008
        lr_anneal_factor_w2v2 = "0.9" # og = 0.9
        lr_decoder = "0.001" #0.0001
        lr_anneal_factor_decoder = "0.8" # og = 0.8
        patience = "0"
        mtl = "False"
        data_dir = "/y/mkperez/speechbrain/AphasiaBank/data/personalization_torre"
        train_data = "train_kansas.csv" # [train_no_kansas.csv, train_all.csv]
        val_data = "val_kansas.csv" # [val_kansas.csv, val_no_kansas.csv]
        test_data = "test_kansas.csv"
        tr_str = train_data.split(".")[0]
        val_str = val_data.split(".")[0]
        personalized_speaker=args.personalized_speaker
        # exp_out_folder = f"results/{exp_name}/freeze-{freeze}/tr-{tr_str}_val-{val_str}"
        exp_out_folder = f"results/{exp_name}/freeze-{freeze}/SD-{personalized_speaker}"




    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('freeze_PLACEHOLDER', f"{freeze}")
        filedata = filedata.replace('grad_acc_PLACEHOLDER', f"{grad_acc}")
        filedata = filedata.replace('batch_size_PLACEHOLDER', f"{batch_size}")
        filedata = filedata.replace('epoch_PLACEHOLDER', f"{epochs}")
        filedata = filedata.replace('lr_w2v2_PLACEHOLDER', f"{lr_w2v2}")
        filedata = filedata.replace('lr_decoder_PLACEHOLDER', f"{lr_decoder}")
        filedata = filedata.replace('mtl_PLACEHOLDER', f"{mtl}")
        filedata = filedata.replace('train_data_PLACEHOLDER', f"{train_data}")
        filedata = filedata.replace('val_data_PLACEHOLDER', f"{val_data}")
        filedata = filedata.replace('test_data_PLACEHOLDER', f"{test_data}")
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_dir}")
        filedata = filedata.replace('exp_out_folder_PLACEHOLDER', f"{exp_out_folder}")
        filedata = filedata.replace('lr_anneal_factor_w2v2_PLACEHOLDER', f"{lr_anneal_factor_w2v2}")
        filedata = filedata.replace('lr_anneal_factor_decoder_PLACEHOLDER', f"{lr_anneal_factor_decoder}")
        filedata = filedata.replace('patience_PLACEHOLDER', f"{patience}")
        filedata = filedata.replace('personalized_speaker_PLACEHOLDER', f"{personalized_speaker}")
        

        with open(outfile,'w') as fout:
            fout.write(filedata)



def create_yaml_PT_FT_fluency(args):
    # args.severity=="PT"

        

    # Create new yaml and return path to generated yaml
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/PT_FT/base_placeholder.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(f"Making YAML: {exp_name} - {sev_arg}")

    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    # copy file over
    shutil.copyfile(base_yaml, outfile)


    # variables
    assert args.severity in ['PT', 'FT']
    assert args.fluency in ['Fluent', 'Non-Fluent']
    data_root="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process"
    data_folder=f"{data_root}/PT_FT-fluency/{args.fluency}"

    model = ['facebook/wav2vec2-large-960h-lv60-self','facebook/hubert-large-ll60k'][0]
    model_str= model.split("/")[-1].split("-")[0]
    freeze = False
    batch_size=3
    grad_acc=16
    train_data=f"<data_folder>/{args.severity}_train.csv"
    dev_data=f"<data_folder>/{args.severity}_dev.csv"
    test_data=f"<data_folder>/test.csv"
    exp_out_folder=f"results/duc_process/PT-FT_fluency/{args.severity}-{args.fluency}/No-LM_{model_str}/freeze-<freeze_wav2vec>"
    

    if args.severity == 'PT':
        epochs=15
        wav2vec2_lr="1.e-4"
        decoder_lr=0.9
    elif args.severity == 'FT':
        epochs=30
        wav2vec2_lr="8.e-5"
        decoder_lr=0.55


    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('model_PLACEHOLDER', f"{model}")
        filedata = filedata.replace('freeze_PLACEHOLDER', f"{freeze}")
        filedata = filedata.replace('batch_size_PLACEHOLDER', f"{batch_size}")
        filedata = filedata.replace('grad_acc_PLACEHOLDER', f"{grad_acc}")
        filedata = filedata.replace('epoch_PLACEHOLDER', f"{epochs}")
        filedata = filedata.replace('train_data_PLACEHOLDER', f"{train_data}")
        filedata = filedata.replace('val_data_PLACEHOLDER', f"{dev_data}")
        filedata = filedata.replace('test_data_PLACEHOLDER', f"{test_data}")
        filedata = filedata.replace('exp_out_folder_PLACEHOLDER', f"{exp_out_folder}")
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_folder}")
        filedata = filedata.replace('w2v_lr_PLACEHOLDER', f"{wav2vec2_lr}")
        filedata = filedata.replace('lr_decoder_PLACEHOLDER', f"{decoder_lr}")
        # filedata = filedata.replace('mtl_PLACEHOLDER', f"{mtl}")
        # filedata = filedata.replace('lr_anneal_factor_w2v2_PLACEHOLDER', f"{lr_anneal_factor_w2v2}")
        # filedata = filedata.replace('lr_anneal_factor_decoder_PLACEHOLDER', f"{lr_anneal_factor_decoder}")
        # filedata = filedata.replace('patience_PLACEHOLDER', f"{patience}")
        # filedata = filedata.replace('personalized_speaker_PLACEHOLDER', f"{personalized_speaker}")
        

        with open(outfile,'w') as fout:
            fout.write(filedata)

def create_yaml_PT_FT(args):
    '''
    Generic
    Warmup PT + FT
    '''
    # Create new yaml and return path to generated yaml
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/PT_FT/base_placeholder.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(f"Making YAML: {exp_name} - {sev_arg}")

    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    # copy file over
    shutil.copyfile(base_yaml, outfile)


    # variables
    data_root="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process"
    data_folder=f"{data_root}"

    model_path = ['facebook/wav2vec2-large-960h-lv60-self','facebook/hubert-large-ll60k','openai/whisper-large-v2'][2]
    model_str = model_path.split("/")[-1].split("-")[0]
    lr_w2v2=1e-6 # for whisper net

    train_data=f"<data_folder>/train.csv"
    dev_data=f"<data_folder>/dev.csv"
    test_data=f"<data_folder>/test.csv"
    exp_out_folder=f"results/duc_process/PT-FT/No-LM_{model_str}/{args.severity}/freeze-<freeze_wav2vec>"


    if args.severity == 'warmup':
        epochs=10
        batch_size=64
        grad_acc=2
        freeze = True
    elif args.severity == 'FT':
        epochs=30
        batch_size=3
        grad_acc=16
        freeze = False


    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('model_PLACEHOLDER', f"{model_path}")
        filedata = filedata.replace('freeze_PLACEHOLDER', f"{freeze}")
        filedata = filedata.replace('batch_size_PLACEHOLDER', f"{batch_size}")
        filedata = filedata.replace('grad_acc_PLACEHOLDER', f"{grad_acc}")
        filedata = filedata.replace('epoch_PLACEHOLDER', f"{epochs}")
        filedata = filedata.replace('train_data_PLACEHOLDER', f"{train_data}")
        filedata = filedata.replace('val_data_PLACEHOLDER', f"{dev_data}")
        filedata = filedata.replace('test_data_PLACEHOLDER', f"{test_data}")
        filedata = filedata.replace('exp_out_folder_PLACEHOLDER', f"{exp_out_folder}")
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_folder}")
        filedata = filedata.replace('w2v_lr_PLACEHOLDER', f"{lr_w2v2}")

        with open(outfile,'w') as fout:
            fout.write(filedata)

def create_yaml_PT_FT_subtype(args):
    # Create new yaml and return path to generated yaml
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/PT_FT/base_placeholder.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(f"Making YAML: {exp_name} - {sev_arg}")

    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    # copy file over
    shutil.copyfile(base_yaml, outfile)


    # variables
    assert args.severity in ['PT', 'FT']
    assert args.subtype in ["Anomic", "Broca", "Conduction", "Global", "TransMotor", "TransSensory", "Wernicke"]
    data_root="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process"
    data_folder=f"{data_root}/PT_FT-subtype/{args.subtype}"

    model = ['facebook/wav2vec2-large-960h-lv60-self','facebook/hubert-large-ll60k'][0]
    model_str= model.split("/")[-1].split("-")[0]
    freeze = False #Change back
    batch_size=9
    grad_acc=16
    train_data=f"<data_folder>/{args.severity}_train.csv"
    dev_data=f"<data_folder>/{args.severity}_dev.csv"
    test_data=f"<data_folder>/test.csv"
    exp_out_folder=f"results/duc_process/PT-FT_subtype/{args.severity}-{args.subtype}/No-LM_{model_str}/freeze-<freeze_wav2vec>"
    

    if args.severity == 'PT':
        epochs=15
        wav2vec2_lr="1.e-4"
        decoder_lr=0.9
    elif args.severity == 'FT':
        epochs=30
        wav2vec2_lr="8.e-5"
        decoder_lr=0.55


    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('model_PLACEHOLDER', f"{model}")
        filedata = filedata.replace('freeze_PLACEHOLDER', f"{freeze}")
        filedata = filedata.replace('batch_size_PLACEHOLDER', f"{batch_size}")
        filedata = filedata.replace('grad_acc_PLACEHOLDER', f"{grad_acc}")
        filedata = filedata.replace('epoch_PLACEHOLDER', f"{epochs}")
        filedata = filedata.replace('train_data_PLACEHOLDER', f"{train_data}")
        filedata = filedata.replace('val_data_PLACEHOLDER', f"{dev_data}")
        filedata = filedata.replace('test_data_PLACEHOLDER', f"{test_data}")
        filedata = filedata.replace('exp_out_folder_PLACEHOLDER', f"{exp_out_folder}")
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_folder}")
        filedata = filedata.replace('w2v_lr_PLACEHOLDER', f"{wav2vec2_lr}")
        filedata = filedata.replace('lr_decoder_PLACEHOLDER', f"{decoder_lr}")
        # filedata = filedata.replace('mtl_PLACEHOLDER', f"{mtl}")
        # filedata = filedata.replace('lr_anneal_factor_w2v2_PLACEHOLDER', f"{lr_anneal_factor_w2v2}")
        # filedata = filedata.replace('lr_anneal_factor_decoder_PLACEHOLDER', f"{lr_anneal_factor_decoder}")
        # filedata = filedata.replace('patience_PLACEHOLDER', f"{patience}")
        # filedata = filedata.replace('personalized_speaker_PLACEHOLDER', f"{personalized_speaker}")
        

        with open(outfile,'w') as fout:
            fout.write(filedata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

    parser.add_argument('-e','--exp_name') 
    parser.add_argument('-s', '--severity')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-p', '--personalized_speaker')
    parser.add_argument('-f', '--fluency')
    parser.add_argument('-t', '--subtype')

    args = parser.parse_args()

    if args.exp_name == "torre_preprocess":
        create_yaml_torre(args)
    elif args.exp_name == "personalization" and args.severity=="last2_w2v":
        create_yaml_torre(args)
    elif args.exp_name == "personalization" and args.severity=="personalize":
        create_yaml_personalization(args)

    elif args.exp_name == "PT_FT" and args.fluency:
        create_yaml_PT_FT_fluency(args)

    elif args.exp_name == "PT_FT" and args.subtype:
        create_yaml_PT_FT_subtype(args)

    elif args.exp_name == "PT_FT" and args.severity in ['warmup', 'FT']:
        create_yaml_PT_FT(args)
