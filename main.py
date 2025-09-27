import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_eval, OOD_Dataset_eval
from utils import reproducibility
from utils import read_metadata
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i=0
    with torch.no_grad():
      for _, batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        target = torch.LongTensor(batch_y).to(device)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        pred = batch_out.max(1)[1] 
        correct += pred.eq(target).sum().item()
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        i=i+1
    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print('\n{} - {} - {} '.format(epoch, str(test_accuracy)+'%', val_loss))
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    batch_size = 10
    num_workers = 4
    print("Batch size: {}, No. workers: {}".format(batch_size, num_workers))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=8)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []
    
    for batch in tqdm(data_loader):
        batch_x, utt_id = batch
        if 'FoR' in save_path:
            utt_id = list(map(lambda x: '/'.join(x.split('/')[-2:]), utt_id))
        else:
            utt_id = list(map(lambda x: x.split('/')[-1], utt_id))
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    for f, ens_cm in zip(fname_list, score_list):
        text_list.append('{} {}'.format(f, ens_cm))
    del fname_list
    del score_list
    with open(save_path, 'a+') as fh:
        fh.write('\n'.join(text_list) + '\n')
    print('Scores saved to {}'.format(save_path))
    del text_list
    fh.close()

def train_epoch(train_loader, model, lr,optim, device, ga):
    num_total = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    i=0
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        batch_x_clean, batch_x_noise, batch_y = batch
        batch_x = torch.cat((batch_x_clean, batch_x_noise), dim=0)
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_out_clean, batch_out_noise = torch.split(batch_out, batch_size//2)

        if ga is not None:
            batch_loss_clean = criterion(batch_out_clean, batch_y)     
            batch_loss_noise = criterion(batch_out_noise, batch_y)     
            batch_loss = (batch_loss_clean + batch_loss_noise) / 2
            optim.ga_backward([batch_loss_clean, batch_loss_noise])
        else:
            batch_loss = criterion(batch_out, torch.cat([batch_y, batch_y], dim=0))
            optim.zero_grad()
            batch_loss.backward()
        optim.step()
        i=i+1
        pbar.set_description(f"Epoch {epoch}: cls_loss {batch_loss.item()}")
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    # Dataset
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASVspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    parser.add_argument('--model', type=str, default='conformer_tcm')
    parser.add_argument('--ga', type=str, default=None)

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--num_mamba_encoders', type=int, default=12, metavar='N',
                    help='number of encoders of the mamba')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    
    #Train
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=4, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    if not os.path.exists('exps'):
        os.mkdir('exps')
    args = parser.parse_args()
    print(args)
    args.track='LA'
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = '{}_w_GA{}'.format(args.model, args.ga)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('exps', model_tag)
    
    print('Model tag: '+ model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
        
    best_save_path = os.path.join(model_save_path, 'best')
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    if args.model == 'conformer_tcm':
        from nets.conformer_tcm import Model
        model = Model(args,device)
    elif args.model == 'aasist':
        from nets.aasist import Model
        model = Model(args,device)
    elif args.model == 'mamba':
        from nets.mamba import Model
        model = Model(args,device)
    else:
        print('Undefined model architechture!')
        exit()
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    if args.ga == 'pcgrad':
        print('Using PCGrad...')
        from grad_align.pcgrad import PCGrad 
        optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay))
    elif args.ga == 'gradvac':
        print('Using GradVac...')
        from grad_align.gradvac import GradVac 
        optimizer = GradVac(torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay))
    elif args.ga == 'cagrad':
        print('Using CAGrad...')
        from grad_align.cagrad import CAGrad 
        optimizer = CAGrad(torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay))
    else:
        print('Without Using Gradient Alignment...')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    # define train dataloader
    label_trn, files_id_train = read_metadata( dir_meta =  os.path.join(args.protocols_path+'LA/{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of training trials',len(files_id_train))
    
    train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn, base_dir = os.path.join(args.database_path+'LA/{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True)
    
    # define validation dataloader
    labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path+'LA/{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of validation trials',len(files_id_dev))

    dev_set = Dataset_train(args,list_IDs = files_id_dev, labels = labels_dev, base_dir = os.path.join(args.database_path+'LA/{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)),
                            algo=0)

    dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False)
    del dev_set,labels_dev

    
    ##################### Training and validation #####################
    num_epochs = args.num_epochs
    not_improving=0
    epoch=0
    bests=np.ones(n_mejores,dtype=float)*float('inf')
    best_metric=float('inf')
    if args.train:
        if not os.path.exists(best_save_path):
            os.mkdir(best_save_path)
        else:
            print('Experiment exists!!!')
            exit()
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
        while not_improving<args.num_epochs:
            print('######## Epoca {} ########'.format(epoch))
            train_epoch(train_loader, model, args.lr, optimizer, device, args.ga)

            val_metric = evaluate_accuracy(dev_loader, model, device, epoch)
            if val_metric<best_metric:
                best_metric=val_metric
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving=0
            else:
                not_improving+=1
            for i in range(n_mejores):
                if bests[i]>val_metric:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests[i]=val_metric
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    break
            print('\n{} - {}'.format(epoch, val_metric))
            print('n-best loss:', bests)
            epoch+=1
            if epoch>100:
                break
        print('Total epochs: ' + str(epoch) +'\n')


    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    ood_eval_tracks=['In-the-Wild', 'FoR-norm-test']
    asvspoof_eval_tracks=['LA', 'DF']
    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)
    if 'In-the-Wild' in ood_eval_tracks:
        file_eval = read_metadata(dir_meta='database/protocols/in_the_wild.eval.txt', is_eval=True)
        eval_set=OOD_Dataset_eval(list_IDs = file_eval, base_dir='/data/spk_corpora/release_in_the_wild/')
        if not os.path.exists('Scores/In-the-Wild/{}.txt'.format(model_tag)):
            produce_evaluation_file(eval_set, model, device, 'Scores/In-the-Wild/{}.txt'.format(model_tag))        
        else:
            print('Score ITW file already exists')
    if 'FoR-norm-test' in ood_eval_tracks:
        file_eval = read_metadata(dir_meta='database/protocols/for_norm_test.txt', is_eval=True)
        eval_set=OOD_Dataset_eval(list_IDs = file_eval, base_dir='/data/spk_corpora/FoR/for-norm/testing/')
        if not os.path.exists('Scores/FoR-norm_test/{}.txt'.format(model_tag)):
            produce_evaluation_file(eval_set, model, device, 'Scores/FoR-norm-test/{}.txt'.format(model_tag))        
        else:
            print('Score FoR file already exists')
    if True:
        for tracks in asvspoof_eval_tracks:
            if not os.path.exists('Scores/{}/{}.txt'.format(tracks, model_tag)):
                prefix      = 'ASVspoof_{}'.format(tracks)
                prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
                prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

                file_eval = read_metadata( dir_meta = os.path.join(args.protocols_path+'{}/{}_cm_protocols/{}.cm.eval.trl.txt'.format(tracks, prefix,prefix_2021)), is_eval=True)
                print('no. of eval trials',len(file_eval))
                eval_set=Dataset_eval(list_IDs = file_eval, base_dir = os.path.join(args.database_path+'{}/ASVspoof2021_{}_eval/'.format(tracks,tracks)),track=tracks)
                produce_evaluation_file(eval_set, model, device, 'Scores/{}/{}.txt'.format(tracks, model_tag))
            else:
                print('Score file already exists')
