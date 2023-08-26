Speech Acoustic Modeling Example
================================
This folder contains examples for speech recognition.

- [lstm_proj.py](lstm.py): Functions for building a LSTM Network with/without projection layer.
- [io_util.py](io_util.py): Wrapper functions for `DataIter` over speech data.
- [train_lstm_proj.py](train_lstm_proj.py): Script for training LSTM acoustic model.
- [decode_mxnet.py](decode_mxnet.py): Script for decoding LSTMP acoustic model.
- [default.cfg](default.cfg): Configuration for training on the `AMI` SDM1 dataset. Can be used as a template for writing other configuration files.
- [python_wrap](python_wrap): C wrappers for Kaldi C++ code, this is built into a .so. Python code that loads the .so and calls the C wrapper functions in `io_func/feat_readers/reader_kaldi.py`.

Connect to Kaldi:
- [decode_mxnet.sh](decode_mxnet.sh): called by Kaldi to decode a acoustic model trained by mxnet (please select the `simple` method for decoding).

A full receipt:
- [run_ami.sh](run_ami.sh): a full receipt to train and decode acoustic model on AMI. It takes features and alignment from Kaldi to train an acoustic model and decode it.

To reproduce the results, use the following steps.

### Build Kaldi

Build Kaldi as **shared libraties** if you have not already done so.

```bash
cd kaldi/src
./configure --shared # and other options that you need
make depend
make
```

### Build Python Wrapper

1. Copy or link the attached `python_wrap` folder to `kaldi/src`.
2. Compile python_wrap/

```
cd kaldi/src/python_wrap/
make
```

### Extract Features and Prepare Frame-level Labels

The acoustic models use *Mel filter-bank* or *MFCC* as input features. It also need to use Kaldi to do force-alignment to generate frame-level labels from the text transcriptions. For example, if you want to work on the `AMI` data `SDM1`. You can run `kaldi/egs/ami/s5/run_sdm.sh`. You will need to do some configuration of paths in `kaldi/egs/ami/s5/cmd.sh` and `kaldi/egs/ami/s5/run_sdm.sh` before you can run the examples. Please refer to Kaldi's document for more details.

The default `run_sdm.sh` script generates the force-alignment labels in their stage 7, and saves the force-aligned labels in `exp/sdm1/tri3a_ali`. The default script generates MFCC features (13-dimensional). You can try training with the MFCC features, or you can create Mel filter bank features by your self. For example, a script like this can be used to compute Mel filter bank features using Kaldi.

```bash
#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# SDM - Signle Distant Microphone
micid=1 #which mic from array should be used?
mic=sdm$micid

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
data_dir=$PWD/data/$mic

# make filter bank data
for dset in train dev eval; do
  steps/make_fbank.sh --nj 48 --cmd "$train_cmd" $data_dir/$dset \
    $data_dir/$dset/log $data_dir/$dset/data-fbank
  steps/compute_cmvn_stats.sh $data_dir/$dset \
    $data_dir/$dset/log $data_dir/$dset/data

  apply-cmvn --utt2spk=ark:$data_dir/$dset/utt2spk \
    scp:$data_dir/$dset/cmvn.scp scp:$data_dir/$dset/feats.scp \
    ark,scp:$data_dir/$dset/feats-cmvn.ark,$data_dir/$dset/feats-cmvn.scp

  mv $data_dir/$dset/feats-cmvn.scp $data_dir/$dset/feats.scp
done
```
Here `apply-cmvn` was for mean-variance normalization. The default setup was applied per speaker. A more common was doing mean-variance normalization for the whole corpus and then feed to the neural networks:
```
 compute-cmvn-stats scp:data/sdm1/train_fbank/feats.scp data/sdm1/train_fbank/cmvn_g.ark
 apply-cmvn --norm-vars=true data/sdm1/train_fbank/cmvn_g.ark scp:data/sdm1/train_fbank/feats.scp ark,scp:data/sdm1/train_fbank_gcmvn/feats.ark,data/sdm1/train_fbank_gcmvn/feats.scp
```
Note that kaldi always try to find features in `feats.scp`. So make sure the normalized features organized as Kaldi way during decoding.

Finally, you need to put the features and labels together in a file so that MXNet can find them. More specifically, for each data set (train, dev, eval), you will need to create a file like `train_mxnet.feats`, will the following contents:

```
TRANSFORM scp:feat.scp
scp:label.scp
```

Here the `TRANSFORM` is the transformation you want to apply to the features. By default we use `NO_FEATURE_TRANSFORM`. The `scp:` syntax is from Kaldi. The `feat.scp` is typically the file from `data/sdm1/train/feats.scp`, and the `label.scp` is converted from the force-aligned labels located in `exp/sdm1/tri3a_ali`. Because the force-alignments are only generated on the training data, we split the training set into 90/10 parts, and use the 1/10 hold-out as the dev set (validation set). The script [run_ami.sh](run_ami.sh) will automatically do the spliting and format the file for MXNet. Please set the path in that script correctly before running. The [run_ami.sh](run_ami.sh) script will actually run the full pipeline including 