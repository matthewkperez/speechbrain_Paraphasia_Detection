#!/usr/bin/env/python3
"""
Adapted from:
Recipe for training a whisper-based ctc ASR system with librispeech.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take only the whisper encoder and add a DNN + CTC to fine-tune.
If you want to use the full whisper system, please refer to the recipe
speechbrain/recipes/LibriSpeech/ASR/transformer/train_with_whisper.py
To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper_encoder.yaml
Authors
 * Titouan Parcollet 2022
 * Rudolf A Braun 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from speechbrain.dataio.dataloader import LoopedLoader
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_']

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        tokens, tokens_lens = batch.tokens
        # print(f"tokens: {tokens}")
        # print(f"word: {batch.wrd}")
        # exit()

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass

        # Encode with Whisper and then DNN
        feats = self.modules.whisper(wavs)
        # print(feats.shape)
        x = self.modules.enc(feats)

        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            # print(f"target_words: {target_words}")
            # print(f"batch.wrd: {batch.wrd}")
            # exit()

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.whisper_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            if should_step:
                self.scaler.unscale_(self.whisper_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    if self.optimizer_step > self.hparams.warmup_steps:
                        # Here we added a warmup to the CTC encoder to make sure that
                        # it does not screw the whisper with too large gradients.
                        self.scaler.step(self.whisper_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    # Here we added a warmup to the CTC encoder to make sure that
                    # it does not screw the whisper with too large gradients.
                    if self.optimizer_step > self.hparams.warmup_steps:
                        self.whisper_optimizer.step()
                    self.model_optimizer.step()
                self.whisper_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_whisper, new_lr_whisper = self.hparams.lr_annealing_whisper(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.whisper_optimizer, new_lr_whisper
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_whisperc": old_lr_whisper,
                },
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

    def init_optimizers(self):
        "Initializes the whisper optimizer and model optimizer"
        self.whisper_optimizer = self.hparams.whisper_opt_class(
            self.modules.whisper.parameters()
        )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "whisper_opt", self.whisper_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()



        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            epoch_counter.update_metric(self.valid_loss)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break



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
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens"],
    )


    print(f"train: {len(train_data.data)} -> {len(train_data.data_ids)} | val: {len(valid_data.data)} -> {len(valid_data.data_ids)} | test: {len(test_data.data)} -> {len(test_data.data_ids)}")
    return train_data, valid_data, test_data

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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



    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )



    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = tokenizer
    # print(f"tokenizer: {len(asr_brain.tokenizer)}")
    # print(f"tokenizer: {len(asr_brain.tokenizer.lab2ind.keys())}")

    # Training
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
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )