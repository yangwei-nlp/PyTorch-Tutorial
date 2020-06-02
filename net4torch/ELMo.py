from allennlp.modules.elmo import Elmo, batch_to_ids

# tutorial：https://github.com/allenai/allennlp/blob/5ba77d44d9c87f1a12ec4ea63aca1d965399090f/docs/tutorials/how_to/elmo.md


options_file = "/home/ai/yangwei/myResources/elmo_pretrain/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "/home/ai/yangwei/myResources/elmo_pretrain/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector

print(2333)