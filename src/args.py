import argparse 

def get_args():
    parser = argparse.ArgumentParser(description='GCMC')
    # dataset
    parser.add_argument('-d','--dataset', choices=['bets','lastfm'], required=True,
                        help='dataset to run experiment on (either lastfm or bets)')
    parser.add_argument('-t', '--transform', choices=['log', 'linear', 'ratings', None], default=None,
                        help='choose data preprocessing')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='linear scaling parameter in transformation')
    parser.add_argument('--eps', type=float, default=10e-04,
                        help='log scaling parameter in transformation')
    # training
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', type=float, default=0.01,
                        help='learning rate to apply for user updates')    
    parser.add_argument('--batch-ratio', type=int, default=0,
                        help='number of batches per epoch')    
    parser.add_argument('--num-neg', type=int, default=2,
                        help='number of negative pairs per positive observation')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='l2 regularisation parameter for optimisation')                            
    # model params
    parser.add_argument('--hidden-dim', type=int, default=500, metavar='H',
                        help='hidden dimension size')                        
    parser.add_argument('-f','--factors', type=int, default=128, metavar='F',
                        help='latent dimension size')    
    parser.add_argument('--drop-prob', type=float, default=0.5,
                        help='node dropout probability')
    parser.add_argument('--rgc-bn', type=int, default=0,
                        help='batch norm for rgc layer') 
    parser.add_argument('--rgc-relu', type=int, default=1,
                        help='relu for rgc layer') 
    parser.add_argument('--dense-bn', type=int, default=0,
                        help='bn for dense layer') 
    parser.add_argument('--dense-relu', type=int, default=1,
                        help='relu for dense layer') 
    parser.add_argument('--bidec-drop', type=int, default=1,
                        help='dropout for dense layer') 
    parser.add_argument('--epoch-per-batch', type=int, default=0,
                        help='dropout for dense layer') 
    # side info
    parser.add_argument('--item-side-info', type=int, default=0,
                        help='use item level side information')
    parser.add_argument('--feat-hidden-size', type=int, default=16, metavar='H',
                        help='feature hidden dimension size')    
    # metrics
    parser.add_argument('--top-n', type=int, default=5, metavar='top_N',
                        help='number of recommendations to make per user - to be evaluated used MAP@N and Recall')
    parser.add_argument('--project', type=str, required=True,
                        help='wandb project name')
    parser.add_argument('--calc-eval', type=int)
    args = parser.parse_args()

    if args.calc_eval == 1:
        args.calc_eval = True
    else:
        args.calc_eval = False
        
    # set paths
    args.root = 'data/'+args.dataset+'/'
    args.test_root = 'data/'+args.dataset+'/test/'
    if args.item_side_info:
        args.side_root = 'data/'+args.dataset+'/side_info/'
    args.hidden_size = [args.hidden_dim, args.factors]

    # set specific transforms
    if args.dataset == 'lastfm':
        args.url = 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz'
        args.file_path = 'lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
        args.cols = ['userId', 'timestamp', 'artistId', 'artist-name','songId', 'song-name']
        # args.num_items = 26307 # train
        args.num_items = 26775
        args.t = 'log'
        args.alpha = 40
        args.eps = 10e-04
        

    return args