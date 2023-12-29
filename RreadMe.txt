Type "python main.py" to run the model.

Before run the model, type "python Dataprocess.py" to obtain WN18RREntTxtWeights.npy and FB15K237EntTxtWeights.npy.

WN18RR Parameter Configuration
############
1. main.py
parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="KGs/WN18RR", nargs="?",
                        help="Which dataset to use with KGs/XXX: ,WN18RR, FB15k-237.")
    parser.add_argument("--embedding_dim", type=int, default=100, nargs="?",
                        help="Number of dimensions in embedding space.")
    parser.add_argument("--num_of_epochs", type=int, default=100, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=512, nargs="?",
                        help="Batch size.")
    parser.add_argument("--input_dropout", type=float, default=0.5, nargs="?",
                        help="Dropout rate for concatenated embeddings.")
    parser.add_argument("--hidden_dropout", type=float, default=0.5, nargs="?",
                        help="Dropout rate for composite embeddings.")
    parser.add_argument("--hidden_width_rate", type=int, default=3, nargs="?",
                        help="How many times wider should be the hidden layer than embeddings.")
    parser.add_argument("--L2reg", type=float, default=.1, nargs="?", help="L2.")
2. model.py
embedding_weights = np.load('WN18RREntTxtWeights.npy', allow_pickle=True)
############

FB15k-237 Parameter Configuration
############
1. main.py
parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="KGs/FB15k-237", nargs="?",
                        help="Which dataset to use with KGs/XXX: ,WN18RR, FB15k-237.")
    parser.add_argument("--embedding_dim", type=int, default=50, nargs="?",
                        help="Number of dimensions in embedding space.")
    parser.add_argument("--num_of_epochs", type=int, default=150, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1000, nargs="?",
                        help="Batch size.")
    parser.add_argument("--input_dropout", type=float, default=0.5, nargs="?",
                        help="Dropout rate for concatenated embeddings.")
    parser.add_argument("--hidden_dropout", type=float, default=0.5, nargs="?",
                        help="Dropout rate for composite embeddings.")
    parser.add_argument("--hidden_width_rate", type=int, default=3, nargs="?",
                        help="How many times wider should be the hidden layer than embeddings.")
    parser.add_argument("--L2reg", type=float, default=.1, nargs="?", help="L2.")
2.model.py
embedding_weights = np.load('FB15K237EntTxtWeights.npy', allow_pickle=True)
############