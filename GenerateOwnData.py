# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import shutil
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry
import tensorflow as tf

from collections import defaultdict

_NC_TRAIN_DATASETS = [[
    "https://github.com/WilliamDunbar/Machine_Translation/blob/master/Data.tar.gz",
    ["train.en", "train.vi"]
]]

_NC_TEST_DATASETS = [[
    "https://github.com/WilliamDunbar/Machine_Translation/blob/master/Data.tar.gz",
    ("tst.en", "tst.vi")
]]


def extract_dummy_tar(tmp_dir, dummy_file_name):
    dummy_file_path = os.path.join(tmp_dir, dummy_file_name)
    tf.compat.v1.logging.info("Extracting file: %s", dummy_file_path)
    tar_dummy = tarfile.open(dummy_file_path,'w:gz')
    print(tar_dummy.getnames())
    tar_dummy.extractall()
    tar_dummy.close()
    # shutil.unpack_archive(dummy_file_path,tmp_dir,"tar.gz")
    tf.compat.v1.logging.info("File %s have been already extracted", dummy_file_name)


def get_filename(dataset):
    return dataset[0][0].split("/")[-1]


@registry.register_problem
class TranslateEnzhSub32k(translate.TranslateProblem):
    """Problem spec for WMT En-De translation, BPE version."""

    # Set the size of the word list generation
    @property
    def vocab_size(self):
        return 32000

        # Use bpe segment words

    # @property
    # def vocab_type(self):
    #    return text_problems.VocabType.TOKEN

    # More than words after the word list representation, None expressed replaced by metacharacters
    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return None

    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def source_vocab_name(self):
        return "vocab.enzh-sub-en.%d" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.enzh-sub-zh.%d" % self.approx_vocab_size

    def get_training_dataset(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        datasets = _NC_TRAIN_DATASETS if train else _NC_TEST_DATASETS
        # You can add other data sets here
        return datasets

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        datasets = self.get_training_dataset(dataset_split)
        for item in datasets:
            dummy_file_name = item[0].split("/")[-1]
            extract_dummy_tar(tmp_dir, dummy_file_name)
            s_file, t_file = item[1][0], item[1][1]
            if not os.path.exists(os.path.join(tmp_dir, s_file)):
                raise Exception("Be sure file '%s' is exists in tmp dir" % s_file)
            if not os.path.exists(os.path.join(tmp_dir, t_file)):
                raise Exception("Be sure file '%s' is exists in tmp dir" % t_file)

        source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
        target_datasets = [[item[0], [item[1][1]]] for item in train_dataset]
        source_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.source_vocab_name,
            self.approx_vocab_size,
            source_datasets,
            file_byte_budget=1e8)
        target_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.target_vocab_name,
            self.approx_vocab_size,
            target_datasets,
            file_byte_budget=1e8)
        tag = "train" if train else "dev"
        filename_base = "wmt_enzh_%sk_sub_%s" % (self.approx_vocab_size, tag)
        data_path = translate.compile_data(tmp_dir, datasets, filename_base)
        return text_problems.text2text_generate_encoded(
            text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                 data_path + ".lang2"),
            source_vocab, target_vocab)

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }