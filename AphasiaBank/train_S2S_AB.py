#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the followering:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import numpy as np
from scipy.io import wavfile
import wave
from tqdm import tqdm
import librosa
import pandas as pd
import math
import os
import sys
import torch
import logging
import speechbrain as sb
# import speechbrain.speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from datasets import load_dataset, load_metric, Audio
import re
import time
from speechbrain.tokenizers.SentencePiece import SentencePiece

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_']
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # print(f"batch.wrd: {batch.wrd}")
        # print(f"batch: {props(batch)}")


        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        x = self.modules.enc(wavs) # (B, T, D)
        enc_out, pred = self.modules.dec(x, tokens_bos, wav_lens, pad_idx=self.hparams.blank_index)
        # print(f"enc_out: {enc_out.shape}") # (B,T,D)
        # print(f"pred: {pred.shape}") # (B,T,D)
        # exit()

        # Output layer for seq2seq log-probabilities (use this only)
        logits = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(logits)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch < self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            # print(f"predicted_words: {predicted_words}")
            # print(f"target_words: {target_words}")
            # exit()
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams,tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    #convert severity_cat to int
    train_data.data = {k:{k_2: (int(v_2) if k_2 == 'severity_cat' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}
    # print(train_data.data)
    # exit()
    if hparams["sorting"] == "ascending":
        if 'tr_speaker' in hparams:
            # print(train_data.data['kansas12a-59'])
            # create numeric speaker_id
            print(f"hparams: {hparams['tr_speaker']}")
            tr_speaker_int = int(re.findall(r'\d+', hparams["tr_speaker"])[0])
            train_data.data = {k:{k_2: (int(re.findall(r'\d+', v_2)[0]) if k_2 == 'spk_id' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}
            # we sort training data to speed up training and get better results.
            train_data = train_data.filtered_sorted(sort_key="duration",
                key_max_value={"duration": hparams["max_length"], 
                    "severity_cat": hparams["max_sev_train"],
                    "spk_id": tr_speaker_int
                },
                key_min_value={"duration": hparams["min_length"], 
                    "severity_cat": hparams["min_sev_train"],
                    "spk_id": tr_speaker_int
                },
            )

        else:
            # we sort training data to speed up training and get better results.
            train_data = train_data.filtered_sorted(sort_key="duration",
                key_max_value={"duration": hparams["max_length"], "severity_cat": hparams["max_sev_train"]},
                key_min_value={"duration": hparams["min_length"], "severity_cat": hparams["min_sev_train"]},

            )

        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True,
            # key_max_value={"duration": hparams["max_length"]}
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder}
    )
    if 'tr_speaker' in hparams:
        valid_data.data = {k:{k_2: (int(re.findall(r'\d+', v_2)[0]) if k_2 == 'spk_id' else v_2) for k_2,v_2 in v.items()} for k,v in valid_data.data.items()}
        # we sort training data to speed up training and get better results.
        valid_data = valid_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"], 
                "spk_id": tr_speaker_int
            },
            key_min_value={"duration": hparams["min_length"], 
                "spk_id": tr_speaker_int
            },
        )
    else:
        valid_data = valid_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"]},
            key_min_value={"duration": hparams["min_length"]}
        )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    if 'tr_speaker' in hparams:
        test_data.data = {k:{k_2: (int(re.findall(r'\d+', v_2)[0]) if k_2 == 'spk_id' else v_2) for k_2,v_2 in v.items()} for k,v in test_data.data.items()}
        # we sort training data to speed up training and get better results.
        test_data = test_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"], 
                "spk_id": tr_speaker_int
            },
            key_min_value={"duration": hparams["min_length"], 
                "spk_id": tr_speaker_int
            },
        )
    else:
        test_data = test_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"]},
            key_min_value={"duration": hparams["min_length"]}
        )

    datasets = [train_data, valid_data, test_data]



    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        # print(f"tokens_bos: {tokens_bos}")
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )


    return (
        train_data,
        valid_data,
        test_data,
    )


def prep_exp_dir(hparams):
    save_folder = hparams['save_folder']
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(f"run_opts: {run_opts}")
    # exit()

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    prep_exp_dir(hparams)


    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=1.0,
    )


    train_data,valid_data,test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    # # We load the pretrained wav2vec2 model
    # if "pretrainer" in hparams.keys():
    #     run_on_main(hparams["pretrainer"].collect_files)
    #     hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = tokenizer.sp
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    tokens = {i:asr_brain.tokenizer.id_to_piece(i) for i in range(asr_brain.tokenizer.get_piece_size())}
    print(f"tokenizer: {tokens}")
    # exit()
    
    # asr_brain.modules = asr_brain.modules.float()
    count_parameters(asr_brain.modules)

    with torch.autograd.detect_anomaly():
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

    # Testing
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_folder"], "wer.txt"
    )
    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_folder"], "cer.txt"
    )
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )

