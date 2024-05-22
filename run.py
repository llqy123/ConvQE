import os
import argparse
import time
import logging
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.knowledge_graph import load_data

from model import ConvQE
from utils import process, TrainDataset, TestDataset
from torch.optim.lr_scheduler import MultiStepLR

import warnings
warnings.filterwarnings("ignore")

class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.data = load_data(self.p.dataset)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels
        self.triplets = process({'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
                                self.num_rels)

        self.p.embed_dim = self.p.k_w * \
            self.p.k_h if self.p.embed_dim is None else self.p.embed_dim  # output dim of gnn
        self.data_iter = self.get_data_iter()

        self.model = ConvQE(self.num_ent, self.num_rels, params=self.p)
        if self.p.gpu >= 0:
            self.model.to("cuda")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.prj_path / 'logs' / self.p.name),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        # pprint(vars(self.p))

    def fit(self):
        save_root = self.prj_path / 'checkpoints'

        if not save_root.exists():
            save_root.mkdir()
        save_path = save_root / (self.p.name + '.pt')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        # if self.p.dataset == "FB15k":
        #     scheduler = MultiStepLR(self.optimizer, milestones=[150, 300, 450], gamma=0.5)
        if self.p.dataset == "FB15k-237":
            scheduler = MultiStepLR(self.optimizer, milestones=[200, 400, 600, 800], gamma=0.4)
        else:
            scheduler = MultiStepLR(self.optimizer, milestones=[200, 500, 800], gamma=0.5)

        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train()
            val_results = self.evaluate('valid')
            # test_results = self.evaluate('test')
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_results = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            # print(
            #     f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            # self.logger.info(
            #     f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Test MRR: {test_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            # self.logger.info(f"hits@1 = {test_results['hits@1']:.5}")
            # self.logger.info(f"hits@3 = {test_results['hits@3']:.5}")
            # self.logger.info(f"hits@10 = {test_results['hits@10']:.5}")
            scheduler.step()
            print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results = self.evaluate('test')
        end = time.time()
        self.logger.info(
            f"MRR: Tail {test_results['left_mrr']:.5}, Head {test_results['right_mrr']:.5}, Avg {test_results['mrr']:.5}")
        self.logger.info(
            f"MR: Tail {test_results['left_mr']:.5}, Head {test_results['right_mr']:.5}, Avg {test_results['mr']:.5}")
        self.logger.info(f"hits@1 = {test_results['hits@1']:.5}")
        self.logger.info(f"hits@3 = {test_results['hits@3']:.5}")
        self.logger.info(f"hits@10 = {test_results['hits@10']:.5}")
        self.logger.info("time ={}".format(end-start))

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            if self.p.gpu >= 0:
                triplets, labels = triplets.to("cuda"), labels.to("cuda")
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred = self.model(subj, rel)  # [batch_size, num_ent]
            loss = self.model.calc_loss(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        return loss

    def evaluate(self, split):
        """
        Function to evaluate the model on validation or test set
        :param split: valid or test, set which data-set to evaluate on
        :return: results['mr']: Average of ranks_left and ranks_right
                 results['mrr']: Mean Reciprocal Rank
                 results['hits@k']: Probability of getting the correct prediction in top-k ranks based on predicted score
                 results['left_mrr'], results['left_mr'], results['right_mrr'], results['right_mr']
                 results['left_hits@k'], results['right_hits@k']
        """

        def get_combined_results(left, right):
            results = dict()
            assert left['count'] == right['count']
            count = float(left['count'])
            results['left_mr'] = round(left['mr'] / count, 5)
            results['left_mrr'] = round(left['mrr'] / count, 5)
            results['right_mr'] = round(right['mr'] / count, 5)
            results['right_mrr'] = round(right['mrr'] / count, 5)
            results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
            results['mrr'] = round(
                (left['mrr'] + right['mrr']) / (2 * count), 5)
            for k in [1, 3, 10]:
                results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
                results[f'right_hits@{k}'] = round(
                    right[f'hits@{k}'] / count, 5)
                results[f'hits@{k}'] = round(
                    (results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        right_result = self.predict(split, 'head')
        res = get_combined_results(left_result, right_result)
        return res

    def predict(self, split='valid', mode='tail'):
        """
        Function to run model evaluation for a given mode
        :param split: valid or test, set which data-set to evaluate on
        :param mode: head or tail
        :return: results['mr']: Sum of ranks
                 results['mrr']: Sum of Reciprocal Rank
                 results['hits@k']: counts of getting the correct prediction in top-k ranks based on predicted score
                 results['count']: number of total predictions
        """
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels) in enumerate(test_iter):
                triplets, labels = triplets.to("cuda"), labels.to("cuda")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred = self.model(subj, rel)
                b_range = torch.arange(pred.shape[0], device="cuda")
                # [batch_size, 1], get the predictive score of obj
                target_pred = pred[b_range, obj]
                # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair
                pred = torch.where(
                    labels.bool(), -torch.ones_like(pred) * 10000000, pred)
                # copy predictive score of obj to new pred
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]  # get the rank of each (sub, rel, obj)
                ranks = ranks.float()
                results['count'] = torch.numel(
                    ranks) + results.get('count', 0)  # number of predictions
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(
                    1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(
                        ranks[ranks <= k]) + results.get(f'hits@{k}', 0)
        return results

    def save_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)

    def load_model(self, path):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])


    def get_data_iter(self):
        """
        get data loader for train, valid and test section
        :return: dict
        """

        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'valid_head': get_data_loader(TestDataset, 'valid_head'),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
            'test_head': get_data_loader(TestDataset, 'test_head'),
            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default='test_run',
                        help='Set run name for saving/restoring models')
    parser.add_argument('--data', dest='dataset', default='FB15k-237',
                        help='Dataset to use, default: FB15k-237')
    parser.add_argument('--temperature', dest='temperature', default=0.007, type=float,
                        help='modify loss function')

    parser.add_argument('--batch', dest='batch_size',
                        default=128, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=800, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',
                        type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345,
                        type=int, help='Seed for randomization')

    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true',
                        help='Whether to use bias in the model')

    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=200, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200,
                        type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')

    parser.add_argument('--Drop1', dest='Drop1', default=0.3, type=float, help='embedding dropout')
    parser.add_argument('--Drop2', dest='Drop2', default=0.3, type=float, help='convolution feature dropout')
    parser.add_argument('--Drop3', dest='Drop3', default=0.2, type=float, help='hidden layer dropout')

    parser.add_argument('--k_w', dest='k_w', default=20,
                        type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10,
                        type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7,
                        type=int, help='ConvE: Kernel size to use')

    parser.add_argument('--gamma', dest='gamma', default=9.0,
                        type=float, help='TransE: Gamma to use')

    parser.add_argument('--rat', action='store_true',
                        default=False, help='random adacency tensors')
    parser.add_argument('--wni', action='store_true',
                        default=False, help='without neighbor information')
    parser.add_argument('--wsi', action='store_true',
                        default=False, help='without self-loop information')
    parser.add_argument('--ss', dest='ss', default=-1,
                        type=int, help='sample size (sample neighbors)')
    parser.add_argument('--nobn', action='store_true',
                        default=False, help='no use of batch normalization in aggregation')
    parser.add_argument('--noltr', action='store_true',
                        default=False, help='no use of linear transformations for relation embeddings')


    # for lte models
    parser.add_argument('--x_ops', dest='x_ops', default="")
    parser.add_argument('--r_ops', dest='r_ops', default="")

    args = parser.parse_args()
    if not args.restore:
        args.name = args.name + 'model'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    runner = Runner(args)
    runner.fit()
