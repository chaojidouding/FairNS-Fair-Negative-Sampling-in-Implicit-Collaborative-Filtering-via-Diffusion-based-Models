import argparse

# MF can be implemented by setting the number of layers of LightGCN to 0.

def parse_args():
    parser = argparse.ArgumentParser(description="AHNS")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="ml-2-types",
                        help="Choose a dataset:[]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="lightgcn",
                        help="Choose a recommender:[lightgcn]")
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048*8, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048*8, help='batch size in evaluation phase')
    # parser.add_argument('--dim', type=int, default=32, help='embedding size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
    parser.add_argument("--simi", type=str, default='ip', help="[ip, cos, ed]")
    parser.add_argument("--context_hops", type=int, default=0, help="hop")
    parser.add_argument("--K", type=int, default=2, help="number of negative samples")

    parser.add_argument("--ns", type=str, default='ahns', help="rns,dns,dns_mn,dens,mix,cns,ahns")
    parser.add_argument("--n_negs", type=int, default=1, help="number of candidate negative")
    parser.add_argument("--p", type=int, default=-2, help="power")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--beta", type=float, default=0.1, help="beta")

    parser.add_argument("--topk", type=int, default=1, help="n for dns_mn")   
    parser.add_argument("--gamma", type=float, default=0.5, help="trade-off parameter of dens")
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20,50]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="weights/", help="output directory for model")

    parser.add_argument('--d_epoch', type=int, default=10, help='diffusion of iteration')
    parser.add_argument('--decay', default=1e-04, type=float, help='weight decay')
    parser.add_argument('--nhid', type=int, default=64, help='hidden size')
    # parser.add_argument('--nhid', type=int, default=32, help='hidden size')
    parser.add_argument('--timesteps', type=int, default=50, help='diffusion of iteration')

    parser.add_argument('--d_weight', type=float, default=0.2, help='weight of diffusion')
    # parser.add_argument('--total_types', type=int, default=2, help='number of types')

    parser.add_argument('--rand_type', type=int, default=1, help='type of random')

    parser.add_argument('--run_type', type=int, default=4, help='type of run')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='d_learning rate')

    parser.add_argument('--w1', type=float, default=0.05, help='w1')
    parser.add_argument('--w2', type=float, default=0.1, help='w2')
    return parser.parse_args()
