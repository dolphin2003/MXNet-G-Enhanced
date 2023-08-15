# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def default_read_content(path):
    with open(path) as ins:
        content = ins.read()
        content = content.replace('\n', ' <eos> ').replace('. ', ' <eos> ')
        return content

def default_build_vocab(path):
    content = default_read_content(path)
    content = content.split(' ')
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    the_vocab[' '] = 0 # put a dummy element here so that len(vocab) is correct
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab

def default_text2id(sentence, the_vocab):
    words = sentence.split(' ')
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words

def default_gen_buckets(sentences, batch_size, the_vocab):
    len_dict = {}
    max_len = -1
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        if len(words) == 0:
            continue
        if len(words) > max_len:
            max_len = len(words)
        if len(words) in len_dict:
            len_dict[len(words)] += 1
        else:
            len_dict[len(words)] = 1
    print(len_dict)

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this    
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets

class ModelParallelBatch(object):
    """Batch used for model parallelism"""
    def __init__(self, data, bucket_key):
        self.data = np.array(data)
        self.bucket_key = bucket_key

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, vocab, buckets, batch_size,
                 init_states, data_name='data', label_name='label',
                 seperate_char=' <eos> ', text2id=None, read_content=None, model_parallel=False):
        super(BucketSentenceIter, self).__init__()

        if text2id == None:
            self.text2id = defau