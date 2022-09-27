# encoding: utf-8
import os
import tqdm
import pickle
import logging
import numpy as np
import pandas as pd
import random
import argparse

from os import path
from os import makedirs
from os.path import join
from itertools import chain
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset
from torch.nn import CrossEntropyLoss, AdaptiveLogSoftmaxWithLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.modeling_bert import ACT2FN
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig


logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


"""args"""
def define_main_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--jid", type=int,
                        default=1,
                        help="job id: 1[default] used for job queue")
    parser.add_argument("--seed", type=int,
                        default=9,
                        help="seed to use: 9[default]")

    parser.add_argument("--flatten", action='store_true',
                        help="enable flattened input, no hierarchical")
    parser.add_argument("--field_ce", type=bool,
                        default=True,
                        help="enable field wise CE")
    parser.add_argument("--mlm", type=bool,
                        default=True,
                        help="masked lm loss; pass it for BERT")
    parser.add_argument("--mlm_prob", type=float,
                        default=0.15,
                        help="mask mlm_probability")

    parser.add_argument("--data_root", type=str,
                        default="./data/",
                        help='root directory for files')
    parser.add_argument("--data_fname", type=str,
                        default="transaction",
                        help='file name of transaction')
    parser.add_argument("--data_extension", type=str,
                        default="",
                        help="file name extension to add to cache")
    parser.add_argument("--vocab_file", type=str,
                        default='vocab.nb',
                        help="cached vocab file")
    parser.add_argument('--user_ids', nargs='+',
                        default=None,
                        help='pass list of user ids to filter data by')
    parser.add_argument("--cached", action='store_true',
                        help='use cached data files')
    parser.add_argument("--nrows", type=int,
                        default=None,
                        help="no of transactions to use")

    parser.add_argument("--output_dir", type=str,
                        default='./outputs',
                        help="path to model dump")
    parser.add_argument("--checkpoint", type=int,
                        default=None,
                        help='set to continue training from checkpoint')
    parser.add_argument("--do_eval", action='store_true',
                        help="enable evaluation flag")
    parser.add_argument("--save_steps", type=int,
                        default=10000,
                        help="set checkpointing")
    parser.add_argument("--num_train_epochs", type=int,
                        default=3,
                        help="number of training epochs")
    parser.add_argument("--stride", type=int,
                        default=5,
                        help="stride for transaction sliding window")
    parser.add_argument("--seq_len", type=int,
                        default=10,
                        help="sequence length for transaction sliding window")
    parser.add_argument("--max_truncate_row", type=int,
                        default=100,
                        help="maximum number of transactions for single user")

    parser.add_argument("--n_layers", type=int,
                        default=2,
                        help="number of transformer blocks")
    parser.add_argument("--field_hs", type=int,
                        default=64,
                        help="hidden size for transaction transformer")
    parser.add_argument("--skip_user", type=bool, 
                        default=True,
                        help="if user field to be skipped or added (default add)")

    return parser



"""utils"""
class ddict(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def random_split_dataset(dataset, lengths, random_seed=20200706):
    # state snapshot
    state = {}
    state['seeds'] = {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(),
        'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

    # seed
    random.seed(random_seed)  # python
    np.random.seed(random_seed)  # numpy
    torch.manual_seed(random_seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)  # torch.cuda

    train_dataset, eval_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, lengths)

    # reinstate state
    random.setstate(state['seeds']['python_state'])
    np.random.set_state(state['seeds']['numpy_state'])
    torch.set_rng_state(state['seeds']['torch_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['seeds']['cuda_state'])

    return train_dataset, eval_dataset, test_dataset


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]



"""Vocabulary"""
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Vocabulary:
    def __init__(self, adap_thres=10000, target_column_name="LABEL"):
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.adap_thres = adap_thres
        self.adap_sm_cols = set()

        self.target_column_name = target_column_name
        self.special_field_tag = "SPECIAL"

        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]

        self.token2id = OrderedDict()  # {field: {token: id}, ...}
        self.id2token = OrderedDict()  # {id : [token,field]}
        self.field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        self.filename = ''  # this field is set in the `save_vocab` method

        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]

    def set_id(self, token, field_name, return_local=False):
        global_id, local_id = None, None

        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        if return_local:
            return local_id

        return global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        global_id, local_id = None, None
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]

        else:
            raise Exception(f"token {token} not found in field: {field_name}")

        if return_local:
            return local_id

        return global_id

    def set_field_keys(self, keys):

        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None

        self.field_keys[self.special_field_tag] = None  # retain the order of columns

    def get_field_ids(self, field_name, return_local=False):
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"field name {field_name} is invalid.")

        selected_idx = 0
        if return_local:
            selected_idx = 1
        return [ids[idx][selected_idx] for idx in ids]

    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)

    def get_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        special_tokens_map = {}
        # TODO : remove the dependency of re-initializing here. retrieve from field_key = SPECIAL
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_



"""Dataset"""
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            print("__name__", __name__, name)
            return super().find_class(__name__, name)     
        except AttributeError:
            print(module, name)
            return super().find_class(module, name)    

class TransactionDataset(Dataset):
    def __init__(self,
                 mlm,
                 user_ids=None,
                 seq_len=10,
                 num_bins=10,
                 cached=True,
                 root="./data/",
                 fname="trans",
                 vocab_dir="checkpoints",
                 fextension="",
                 nrows=None,
                 flatten=False,
                 stride=5,
                 max_truncate_row=100,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=False,
                 vocab=None):

        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.user_ids = user_ids
        self.return_labels = return_labels
        self.skip_user = skip_user

        self.mlm = mlm
        self.trans_stride = stride

        self.flatten = flatten

        self.vocab = Vocabulary(adap_thres)
        self.seq_len = seq_len
        self.max_truncate_row = max_truncate_row
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.unique_user_ids = []
        self.labels = []
        self.window_label = []

        self.ncols = None
        self.num_bins = num_bins
        self.encode_data()
        file_name = path.join(vocab_dir, f'vocab{self.fextension}.nb')
        if self.cached and vocab is not None:
            self.vocab = vocab
        else:
            self.init_vocab()
            self.save_vocab(vocab_dir)
            with open(f'{file_name}.pkl', 'wb') as fout:
                pickle.dump(self.vocab, fout) 
        self.prepare_samples()

    def __getitem__(self, index):
        return_user_id = self.unique_user_ids[index]
        if self.flatten:
            return_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.data[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            return_data = (return_data, torch.tensor(self.labels[index], dtype=torch.long))

        return return_data, return_user_id

    def __len__(self):
        return len(self.data)

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'vocab{self.fextension}.nb')
        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def tramtEncoder(X):
        # amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        # HACK: 
        amt = X.astype(float)
        return pd.DataFrame(amt)

    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs

    def user_level_data(self):
        fname = path.join(self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels = [], []

        if self.cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]
            unique_users = pd.DataFrame(cached_data["unique_users"], columns=["CUST_NO"])
            log.info(f"filtering cached user level data by given user ids")
            if self.user_ids is not None and self.user_ids.any():
                unique_users = unique_users[unique_users["CUST_NO"].isin(self.user_ids)]
                trans_data = np.array(trans_data)[unique_users.index].tolist()
                trans_labels = np.array(trans_labels)[unique_users.index].tolist()
        else:
            unique_users = self.trans_table["CUST_NO"].unique()
            columns_names = list(self.trans_table.columns)

            for user in tqdm.tqdm(unique_users):
                user_data = self.trans_table.loc[self.trans_table["CUST_NO"] == user]
                # HACK: truncated rows
                user_data = user_data[:self.max_truncate_row]

                user_trans, user_labels = [], []
                for idx, row in user_data.iterrows():
                    row = list(row)

                    # assumption that user is first field
                    skip_idx = 1 if self.skip_user else 0

                    user_trans.extend(row[skip_idx:-1])
                    user_labels.append(row[-1])

                trans_data.append(user_trans)
                trans_labels.append(user_labels)

            if self.skip_user:
                columns_names.remove("CUST_NO")

            with open(fname, 'wb') as cache_file:
                pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names, "unique_users": unique_users}, cache_file)

        self.unique_users = unique_users

        # convert to str
        return trans_data, trans_labels, columns_names

    def format_trans(self, trans_lst, column_names):
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            if self.mlm:  # and self.flatten:  # only add [SEP] for BERT + flatten scenario
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        trans_data, trans_labels, columns_names = self.user_level_data()

        log.info("creating transaction samples with vocab")
        pad_id = self.vocab.get_id(self.vocab.pad_token, special_token=True)
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]
            seq_len_res = self.seq_len - len(user_row_ids)
            if seq_len_res > 0:
                ids = list(chain(*user_row_ids))
                ids += list(chain(*([[pad_id] * len(user_row_ids[0])] * seq_len_res)))
                self.data.append(ids)
                self.unique_user_ids.append(self.unique_users.loc[user_idx, "CUST_NO"])
            else:
                for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                    ids = user_row_ids[jdx:(jdx + self.seq_len)]
                    ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                    self.data.append(ids)
                    self.unique_user_ids.append(self.unique_users.loc[user_idx, "CUST_NO"])
            
            if seq_len_res > 0:
                fraud = 0
                if len(np.nonzero(user_labels)[0]) > 0:
                    fraud = 1
                self.window_label.append(fraud)
                
                user_labels.extend([-100] * seq_len_res)
                self.labels.append(user_labels)
            else:
                for jdx in range(0, len(user_labels) - self.seq_len + 1, self.trans_stride):
                    labels = user_labels[jdx:(jdx + self.seq_len)]
                    self.labels.append(labels)

                    fraud = 0
                    if len(np.nonzero(labels)[0]) > 0:
                        fraud = 1
                    self.window_label.append(fraud)

        assert len(self.data) == len(self.labels)

        '''
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 2 + (1 if self.mlm else 0)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.user_ids is not None and self.user_ids.any():
            log.info(f'Filtering data by user ids list: {self.user_ids}...')
            data = data[data['CUST_NO'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def init_vocab(self):
        column_names = list(self.trans_table.columns)
        if self.skip_user:
            column_names.remove("CUST_NO")

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = self.trans_table[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            log.info(f"column : {column}, vocab size : {vocab_size}")

            if vocab_size > self.vocab.adap_thres:
                log.info(f"\tsetting {column} for adaptive softmax")
                self.vocab.adap_sm_cols.add(column)

    def encode_data(self):
        # load data
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")
        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        # encode data
        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        ## TR_CD
        log.info("nan resolution.")
        data = data[~data["TR_CD"].isnull()]  # HACK: drop examples whose TR_CD are None
        
        # continuous features
        ## TR_AMT
        data['TR_AMT'] = self.tramtEncoder(data['TR_AMT'])
        sub_columns = ['TR_AMT']
        ## min-max scaling
        log.info("label-fit-transform.")
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data
        ## TR_DAT
        log.info("date fit transform")
        ## encode time
        tr_dat_fit, tr_dat = self.label_fit_transform(data['TR_DAT'].values.reshape(-1, 1), enc_type="time")
        self.encoder_fit['TR_DAT'] = tr_dat_fit
        data['TR_DAT'] = tr_dat
        
        # discrete features
        ## TR_DAT
        log.info("date quant transform")
        coldata = np.array(data['TR_DAT'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['TR_DAT'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["TR-DAT-Quant"] = [bin_edges, bin_centers, bin_widths]
        # TR_AMT
        log.info("amount quant transform")
        coldata = np.array(data['TR_AMT'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['TR_AMT'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["TR-AMT-Quant"] = [bin_edges, bin_centers, bin_widths]

        # dump data
        ## dump encoded data
        columns_to_select = ['CUST_NO',
                             'TR_DAT',
                             'TR_AMT',
                             'LABEL']
        data = data.reset_index()
        self.trans_table = data[columns_to_select]
        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))
        ## dump encoder
        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))

'''CollatorForLanguageModeling'''
class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        sz = batch.shape
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs.view(sz), "lm_labels": labels.view(sz)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = [example[0] for example in examples]
        user_ids = [example[1] for example in examples]
        batch = self._tensorize_batch(input_ids)
        
        return {"input_ids": batch, "lm_labels": batch, "user_ids": user_ids}



"""Model"""
class TabFormerBaseModel(PreTrainedModel):
    def __init__(self, hf_model, tab_embeddings, config):
        super().__init__(config)

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerHierarchicalLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, n_layers=None, field_hidden_size=768):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size = field_hidden_size if flatten else (field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          n_layers=n_layers,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForMaskedLM(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = TabFormerBertForMaskedLM(self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            model = TabFormerHierarchicalLM(self.config, self.vocab)

        return model

'''Tokenizer'''
class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ):

        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token)

'''TabFormerBertModel'''
class TabFormerBertConfig(BertConfig):
    def __init__(
        self,
        flatten=True,
        ncols=12,
        vocab_size=30522,
        field_hidden_size=64,
        hidden_size=768,
        num_attention_heads=12,
        pad_token_id=0,
        n_layers=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads=num_attention_heads
        self.n_layers = n_layers

class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class TabFormerBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, vocab):
        super().__init__(config)

        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        if not self.config.flatten:
            output_sz = list(sequence_output.size())
            expected_sz = [output_sz[0], output_sz[1]*self.config.ncols, -1]
            sequence_output = sequence_output.view(expected_sz)
            masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1)

        prediction_scores = self.cls(sequence_output) # [bsz * seqlen * vocab_sz]

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # masked_lm_labels  : [bsz x seqlen]

        total_masked_lm_loss = 0

        seq_len = prediction_scores.size(1)
        # TODO : remove_target is True for card
        field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=False)
        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
            masked_lm_labels_field = masked_lm_labels[:, col_ids]
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(global_ids=masked_lm_labels_field,
                                                                          what_to_get='local_ids')

            nfeas = len(global_ids_field)
            loss_fct = self.get_criterion(field_name, nfeas, prediction_scores.device)

            masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)),
                                            masked_lm_labels_field_local.view(-1))

            total_masked_lm_loss += masked_lm_loss_field

        return (total_masked_lm_loss,) + outputs

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):

        if fname in self.vocab.adap_sm_cols:
            if not cutoffs:
                cutoffs = [int(vs/15), 3*int(vs/15), 6*int(vs/15)]

            criteria = CustomAdaptiveLogSoftmax(in_features=vs, n_classes=vs, cutoffs=cutoffs, div_value=div_value)

            return criteria.to(device)
        else:
            return CrossEntropyLoss()

class TabFormerBertModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        return sequence_output

'''Criterion'''
class CustomAdaptiveLogSoftmax(AdaptiveLogSoftmaxWithLoss):
    def __init__(
            self, ignore_index=-100,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        '''
            handles ignore index = -100;
            removes all targets which are masked from input and target
        '''
        consider_indices = (target != self.ignore_index)
        input = input[consider_indices, :]
        target = target[consider_indices]

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = self.head(input)
        head_logprob = log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        return loss

'''Hierarchical TabFormer'''
class TabFormerConcatEmbeddings(nn.Module):
    """TabFormerConcatEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns
               - `sparse=True` in `nn.Embedding` speeds up gradient computation for large vocabs

        Args:
            config.ncols
            config.vocab_size
            config.hidden_size

        Inputs:
            - **input_ids** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output'**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)
        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

        self.hidden_size = config.hidden_size
        self.field_hidden_size = config.field_hidden_size

    def forward(self, input_ids):
        input_shape = input_ids.size()

        embeds_sz = list(input_shape[:-1]) + [input_shape[-1] * self.field_hidden_size]
        inputs_embeds = self.lin_proj(self.word_embeddings(input_ids).view(embeds_sz))

        return inputs_embeds


class TabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns

        Args:
            config.ncols
            config.num_layers (int): Number of transformer layers
            config.vocab_size
            config.hidden_size
            config.field_hidden_size

        Inputs:
            - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()

        if not hasattr(config, 'num_layers'):
            config.num_layers = 1
        if not hasattr(config, 'nhead'):
            config.nhead = 8

        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.field_hidden_size, nhead=config.nhead,
                                                   dim_feedforward=config.field_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

    def forward(self, input_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())

        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2]+[-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds



from torch.utils.data import SequentialSampler, DataLoader

"""Main"""
def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    #==================================arguments===========================
    args.batch_size = 256
    args.checkpoint = 3
    total_data = pd.read_csv(os.path.join(args.data_root, f'{args.data_fname}.csv'))
    test_data = total_data[total_data["LABEL"].isnull()]
    # args.user_ids = test_data["CUST_NO"].unique()
    
    user_model_checkpoint = os.path.join(f'checkpoint-{args.checkpoint}')
    checkpoint = os.path.join(args.output_dir, user_model_checkpoint)
    path_to_save_dir = os.path.join(
        args.output_dir, f'checkpoint-{args.checkpoint}-eval.csv'
    )
    logger.info(f'Saved weights loaded from: {os.path.join(args.output_dir, user_model_checkpoint)}')

    #=================================dataset==============================
    file_name = path.join(args.output_dir, f'vocab{args.data_extension}.nb')
    if path.isfile(file_name):
        with open(f'{file_name}.pkl', 'rb') as fin:
            vocab = pickle.load(fin)
    eval_dataset = TransactionDataset(root=args.data_root,
                                    fname=args.data_fname,
                                    fextension=args.data_extension,
                                    vocab_dir=args.output_dir,
                                    nrows=args.nrows,
                                    user_ids=args.user_ids,
                                    mlm=args.mlm,
                                    cached=True,
                                    stride=args.stride,
                                    seq_len=args.seq_len,
                                    max_truncate_row=args.max_truncate_row,
                                    flatten=args.flatten,
                                    return_labels=False,
                                    skip_user=args.skip_user,
                                    vocab=vocab)
    
    # `TabFormerBertLM` have the ensemble of `config` and `vocab`
    tab_net = TabFormerBertLM(eval_dataset.vocab.get_special_tokens(),
                            vocab=eval_dataset.vocab,
                            field_ce=args.field_ce,
                            flatten=args.flatten,
                            ncols=eval_dataset.ncols,
                            n_layers=args.n_layers,
                            field_hidden_size=args.field_hs
                            )
    data_collator = TransDataCollatorForLanguageModeling(tokenizer=tab_net.tokenizer, mlm=False)  # `mlm=False` for inference
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                sampler=eval_sampler,
                                batch_size=args.batch_size, 
                                collate_fn=data_collator.collate_batch,
                                drop_last=False
                                )
    
    # model = tab_net.model.from_pretrained(pretrained_model_name_or_path=checkpoint, vocab=eval_dataset.vocab)
    tab_net.model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
    model = tab_net.model
    #================================evaluation============================
    log.info(f"evaluation")
    model.eval()
    embeds = []
    for batch in tqdm.tqdm(eval_dataloader, desc="Iteration"):
        input_ids, lm_labels, user_ids = batch["input_ids"], batch["lm_labels"], batch["user_ids"]
        _, field_embed = model(input_ids=input_ids, masked_lm_labels=lm_labels)
        df_embed = pd.DataFrame(field_embed.mean(1).detach().numpy())
        df_embed = pd.concat([df_embed, pd.DataFrame(user_ids, columns=["user_id"])], axis=1)
        df_embed_agg = df_embed.groupby("user_id").agg("mean")
        embeds.append(df_embed_agg)  # mean embedding
        
    embeds_to_write = pd.concat(embeds, axis=0)
    embeds_to_write.to_csv(path_to_save_dir)



if __name__ == '__main__':
    parser = define_main_parser()
    opts = parser.parse_args([])
    opts.log_dir = join(opts.output_dir, "logs")
    print(opts, '\n')

    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    main(opts)
