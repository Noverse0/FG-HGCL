import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from model.loader import DatasetLoader
from model.models import FG_HGCL, HGNN
from model.evaluation import linear_evaluation
from model.utils import fix_seed, plot_tsne


def align_loss(z1, z2, alpha=2):
    # z1과 z2 사이의 차이를 계산하고, L2 노름을 취한 후 제곱합니다.
    return torch.mean(torch.norm(z1 - z2, dim=1) ** alpha).item()


def uniform_loss(z1, z2, t=1.0):
    # z1과 z2 사이의 거리에 대한 음의 지수 함수를 계산합니다.
    pairwise_distances = torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=2, p=2)
    exponential_distances = torch.exp(-t * pairwise_distances)
    
    # 이를 평균내고 로그를 취합니다.
    return -torch.log(torch.mean(exponential_distances) + 1e-15).item()  # 1e-15는 수치 안정성을 위해 추가됩니다.

def train(data, model, optimizer, params, epoch, seed):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Encoder
    n1, e1 = model(data.features, data.hyperedge_index, params['dropout_rate'])
    n1, e1 = model.node_projection(n1), model.edge_projection(e1)

    # Contrastive Loss
    n2, e2 = model(data.features, data.hyperedge_index, 0.0, params['noise_std'])
    n2, e2 = model.node_projection(n2), model.edge_projection(e2)
    
    loss_n = model.calulate_loss(n1, n2, data.overlab_hyperedge, params['n_w_wp'], params['n_w_wn'], params['tau_n'], batch_size=params['batch_size1'], detail=True)
    loss_g = model.calulate_loss(e1, e2, data.overlab_hypernode, params['g_w_wp'], params['g_w_wn'], params['tau_g'], batch_size=params['batch_size2'], detail=True)
    
    loss = loss_n + params['w_g'] * loss_g
    
    loss.backward()
    optimizer.step()
    
    # Print t-SNE plot
    if epoch == params['epochs']:
        # plot_tsne(n1, data.labels, dir="tsne_figure", file_name=f'{args.dataset}_{seed}_fghgcl.pdf')
    
        print('seed', seed)
        print('loss_align_n = ', align_loss(n1, n2, alpha=2))
        print('loss_uniform_n = ', uniform_loss(n1, n2, t=2))
    
    return loss.item(), model

def node_classification_eval(model, data, params, num_splits=20):
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)
    
    lr = params['eval_lr']
    max_epoch = params['eval_epochs']
    
    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
            
    return accs


def main():
    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    
    print(params)
    data = DatasetLoader().load(args.dataset).to(args.device)
    accs = []
    gpu_memory_allocated = []
    gpu_max_memory_allocated = []
    for seed in range(args.num_seeds):
        # Reset the peak memory stats at the beginning of each iteration
        torch.cuda.reset_peak_memory_stats(args.device)
        
        fix_seed(seed)
        encoder = HGNN(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = FG_HGCL(encoder, params['proj_dim'], args.device).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss, model = train(data, model, optimizer, params, epoch, seed)
        
        # At the end of each seed iteration, record the memory usage
        gpu_memory_allocated.append(torch.cuda.memory_allocated(args.device))
        gpu_max_memory_allocated.append(torch.cuda.max_memory_allocated(args.device))

        # evaluation
        acc = node_classification_eval(model, data, params)
        accs.append(acc)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
            f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}')

    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')
    
    # Calculate and print the average GPU memory usage
    avg_gpu_memory_allocated = sum(gpu_memory_allocated) / len(gpu_memory_allocated)
    avg_gpu_max_memory_allocated = sum(gpu_max_memory_allocated) / len(gpu_max_memory_allocated)
    print(f'Average allocated GPU memory: {avg_gpu_memory_allocated / (1024 ** 2):.2f} MB')
    print(f'Average peak allocated GPU memory: {avg_gpu_max_memory_allocated / (1024 ** 2):.2f} MB')    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TriCL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                 'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40'])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    main()
    