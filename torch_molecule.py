import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np

import time
from collections import deque

import gym
import gym_molecule

import os
import copy
import sys

class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)

class GCNPolicy(nn.Module):
    def __init__(self, out_channels=64, stop_shift=-3, atom_type_num=9,in_channels=9, edge_type=3):
        super(GCNPolicy, self).__init__()
        self.stop_shift = stop_shift
        self.atom_type_num = atom_type_num
        self.emb = nn.Linear(in_channels, 8)
        self.ac_real = np.array([])
        
        self.d_gcn1 = nn.Linear(8, out_channels, bias=False)
        self.d_gcn2 = nn.Linear(out_channels, out_channels, bias=False)
        self.d_gcn3 = nn.Linear(out_channels, out_channels, bias=False)
        
        self.g_gcn1 = nn.Linear(8, out_channels, bias=False)
        self.g_gcn2 = nn.Linear(out_channels, out_channels, bias=False)
        self.g_gcn3 = nn.Linear(out_channels, out_channels, bias=False)
        
        self.linear_stop1 = nn.Linear(out_channels, out_channels, bias=False)
        self.linear_stop2 = nn.Linear(out_channels, 2)

        self.linear_first1 = nn.Linear(out_channels, out_channels)
        self.linear_first2 = nn.Linear(out_channels, 1)

        self.linear_second1 = nn.Linear(2*out_channels, out_channels)
        self.linear_second2 = nn.Linear(out_channels, 1)

        self.linear_edge1 = nn.Linear(2*out_channels, out_channels)
        self.linear_edge2 = nn.Linear(out_channels, edge_type)

        self.value1 = nn.Linear(out_channels, out_channels, bias=False)
        self.value2 = nn.Linear(out_channels, 1)

    def mask_emb_len(self, emb_node, mask_len, fill):
        '''
        在结点嵌入emb中，只留前mask_len个结点的特征，
        将之后其他结点的特征全部置为fill
        emb_node: Tensor
        mask_len: int
        fill: int
        '''
        node_num = emb_node.shape[-2]
        v_size = mask_len.tile((1,node_num))
        seq_range = torch.arange(0, node_num).tile(v_size.shape[0],1)
        mask = seq_range>=v_size
        mask = mask.unsqueeze(-1).expand(emb_node.shape)
        return emb_node.masked_fill_(mask,fill)

    def set_ac_real(self, ac_real):
        self.ac_real = ac_real

    def forward(self, adj, node):
        stop_shift = self.stop_shift
        atom_type_num = self.atom_type_num
        self.adj = torch.Tensor(adj)
        self.node = torch.Tensor(node)
        if self.adj.dim() == 3:
            self.adj = self.adj.unsqueeze(0)
        if self.node.dim() == 3:
            self.node = self.node.unsqueeze(0)

        ob_node = self.emb(self.node)
        emb_node = F.relu(self.g_gcn1(torch.einsum("bijk,bikl->bijl",self.adj,ob_node.tile((1,self.adj.shape[1],1,1)))))
        emb_node = torch.mean(emb_node,1).unsqueeze(1)
        emb_node = F.relu(self.g_gcn2(torch.einsum("bijk,bikl->bijl",self.adj,emb_node.tile((1,self.adj.shape[1],1,1)))))
        emb_node = torch.mean(emb_node,1).unsqueeze(1)
        emb_node = F.relu(self.g_gcn3(torch.einsum("bijk,bikl->bijl",self.adj,emb_node.tile((1,self.adj.shape[1],1,1)))))
        emb_node = torch.mean(emb_node,1)
        #(B,n,n) * (B,n,f) -> (B,n,f)
        
        seq_range = torch.arange(0, emb_node.shape[-2])
        ### 1.计算ob中有效node的个数
        ob_len = torch.sum(torch.BoolTensor(torch.sum(self.node,-1)>0),-1)
        ob_len_first = ob_len - atom_type_num
        emb_node = self.mask_emb_len(emb_node, ob_len, 0)

        ### 2.预测停止动作
        emb_stop = F.relu(self.linear_stop1(emb_node))
        self.logits_stop = torch.sum(emb_stop,1) #(B,1,f)
        self.logits_stop = self.linear_stop2(self.logits_stop) #(B,1,2)
        
        # 分类分布中认为1是停止，但不能让它停止的过早，导致生成分子过简单 stop_shift一个负数
        stop_shift = torch.Tensor([[0, stop_shift]])
        pd_stop = D.Categorical(logits=self.logits_stop + stop_shift)
        ac_stop = pd_stop.sample() #(B,1)
        ac_stop = ac_stop.unsqueeze(-1)

        ### 3.1 选第一个有效点(已在分子图中的)
        self.logits_first = F.relu(self.linear_first1(emb_node)) #(B,n,f)
        self.logits_first = self.linear_first2(emb_node).squeeze(-1) #(B,n)
        # 保证选不到无效点
        self.logits_first = self.logits_first.masked_fill(seq_range.expand(self.logits_first.shape)>=ob_len_first.expand(self.logits_first.shape),-10000)
        pd_first = D.Categorical(logits=self.logits_first)
        ac_first = pd_first.sample()
        ac_first = ac_first.unsqueeze(-1) #(B,1)
        # 只留选中结点的emb (B,f)
        emb_first = torch.sum(emb_node.masked_fill(seq_range.unsqueeze(-1).expand(emb_node.shape) != ac_first.squeeze(0).unsqueeze(-1).expand(emb_node.shape),0),-2)      
        # 专家网络ground truth action
        if self.ac_real.size>0:
            ac_first_real = torch.Tensor(self.ac_real[:,0])
            ac_first_real = ac_first_real.unsqueeze(-1)
            emb_first_real = torch.sum(emb_node.masked_fill(seq_range.unsqueeze(-1).expand(emb_node.shape) != ac_first_real.squeeze(0).unsqueeze(-1).expand(emb_node.shape),0),-2) 

        ### 3.2 选第二个点
        emb_cat = torch.cat((emb_first.unsqueeze(-2).expand(emb_node.shape),emb_node), -1) #(B,n,2f)
        self.logits_second = F.relu(self.linear_second1(emb_cat)) #(B,n,f)
        self.logits_second = self.linear_second2(self.logits_second) #(B,n,1)
        self.logits_second = self.logits_second.squeeze(-1)
        self.logits_second = self.logits_second.masked_fill(ac_first.expand(self.logits_second.shape) == seq_range.unsqueeze(0).expand(self.logits_second.shape), -10000)
        self.logits_second = self.logits_second.masked_fill(seq_range.expand(self.logits_second.shape)>=ob_len.expand(self.logits_second.shape),-10000)

        pd_second = D.Categorical(logits=self.logits_second)
        ac_second = pd_second.sample()
        ac_second = ac_second.unsqueeze(-1)
        # 只留选中结点的emb (B,f)
        emb_second = torch.sum(emb_node.masked_fill(seq_range.unsqueeze(-1).expand(emb_node.shape) != ac_second.squeeze(0).unsqueeze(-1).expand(emb_node.shape),0),-2) 

        # groundtruth
        if self.ac_real.size>0:
            emb_cat = torch.cat((emb_first_real.unsqueeze(-2).expand(emb_node.shape),emb_node), -1) #(B,n,2f)
            self.logits_second_real = F.relu(self.linear_second1(emb_cat))
            self.logits_second_real = self.linear_second2(self.logits_second_real).squeeze(-1)
            ac_second_real = torch.Tensor(self.ac_real[:,1])
            ac_second_real = ac_second_real.unsqueeze(-1)
            emb_second_real = torch.sum(emb_node.masked_fill(seq_range.unsqueeze(-1).expand(emb_node.shape) != ac_second_real.squeeze(0).unsqueeze(-1).expand(emb_node.shape),0),-2)

        ### 3.3 预测边类型
        emb_cat = torch.cat((emb_first,emb_second),-1) #(B,2f)
        self.logits_edge = F.relu(self.linear_edge1(emb_cat)) #(B,f)
        self.logits_edge = self.linear_edge2(self.logits_edge) #(B,e)
        pd_edge = D.Categorical(logits = self.logits_edge)
        ac_edge = pd_edge.sample()
        ac_edge = ac_edge.unsqueeze(-1)

        #groundtruth
        if self.ac_real.size>0:
            emb_cat = torch.cat((emb_first_real, emb_second_real), -1)
            self.logits_edge_real = F.relu(self.linear_edge1(emb_cat))
            self.logits_edge_real = self.linear_edge2(self.logits_edge_real)
        
        ### 4. 预测状态价值
        self.vpred = F.relu(self.value1(emb_node))
        self.vpred = torch.max(self.vpred,1).values #(B,1,f)
        self.vpred = self.value2(self.vpred)

        self.ac = torch.cat((ac_first,ac_second,ac_edge,ac_stop),-1)
        self.pd = None
        if self.ac_real.size>0:
            self.pd = {"first": D.Categorical(logits=self.logits_first), "second": D.Categorical(logits=self.logits_second_real), 
                        "edge": D.Categorical(logits=self.logits_edge_real), "stop": D.Categorical(logits=self.logits_stop)}
            self.ac_real = np.array([])
        return self.ac, self.vpred

    def logp(self, ac):
        ac = torch.LongTensor(ac)
        if self.pd != None: 
            return self.pd["first"].log_prob(ac[:,0]) + self.pd["second"].log_prob(ac[:,1])\
                 + self.pd["edge"].log_prob(ac[:,2]) + self.pd["stop"].log_prob(ac[:,3])
        else:
            return None
    
    def entorpy(self):
        result = None
        if self.pd != None:
            result =  self.pd["first"].entropy() + self.pd["second"].entropy()\
                 + self.pd["edge"].entropy() + self.pd["stop"].entropy()
        return result

    def kl(self, other_pd):
        result = None
        if self.pd != None and other_pd != None:
            result = D.kl_divergence(self.pd["first"], other_pd["first"]) + D.kl_divergence(self.pd["second"], other_pd["second"])\
                + D.kl_divergence(self.pd["edge"], other_pd["edge"]) + D.kl_divergence(self.pd["stop"], other_pd["stop"])
        return result

class Discriminator(nn.Module):
    '''
    判断ob是否为生成的
    '''
    def __init__(self, in_channels=9, out_channels=64):
        super(Discriminator, self).__init__()
        self.emb = nn.Linear(in_channels, 8, bias=False)

        self.gcn1 = nn.Linear(8, out_channels, bias=False)
        self.gcn2 = nn.Linear(out_channels, out_channels, bias=False)
        self.gcn3 = nn.Linear(out_channels, out_channels, bias=False)

        self.linear1 = nn.Linear(out_channels, out_channels, bias=False)
        self.linear2 = nn.Linear(out_channels, 1)

    def forward(self, adj, node):
        self.node = torch.Tensor(node)
        self.adj = torch.Tensor(adj)
        if self.adj.dim() == 3:
            self.adj = self.adj.unsqueeze(0)
        if self.node.dim() == 3:
            self.node = self.node.unsqueeze(0)
        ob_node = self.emb(self.node)
        emb_node = F.relu(self.gcn1(torch.einsum("bijk,bikl->bijl",self.adj,ob_node.tile((1,self.adj.shape[1],1,1)))))
        emb_node = torch.mean(emb_node,1).unsqueeze(1)
        emb_node = F.relu(self.gcn2(torch.einsum("bijk,bikl->bijl",self.adj,emb_node.tile((1,self.adj.shape[1],1,1)))))
        emb_node = torch.mean(emb_node,1).unsqueeze(1)
        emb_node = F.relu(self.gcn3(torch.einsum("bijk,bikl->bijl",self.adj,emb_node.tile((1,self.adj.shape[1],1,1)))))
        emb_node = torch.mean(emb_node,1).unsqueeze(1)

        emb_node = F.relu(self.linear1(emb_node)) #(B,n,f)
        emb_graph = torch.sum(emb_node, 1) #(B,f)
        logit = self.linear2(emb_graph) #(B,1)
        pred = F.sigmoid(logit)

        return pred,logit

# 用策略pi生成一个连续时间段的ob与ac
class traj_segment_generator:
    def __init__(self, pi, env, horizon, dis_step, dis_final, step_ratio=1, final_ratio=1):
        self.pi = pi
        self.env = env
        self.horizon = horizon
        self.dis_step = dis_step
        self.dis_final = dis_final
        self.ac_real = np.array([])
        self.step_ratio = step_ratio
        self.final_ratio = final_ratio
    
    def set_ac_real(self, ac_real):
        self.ac_real = ac_real
    
    def get_generator(self, name="test"):
        env = self.env
        horizon = self.horizon
        t = 0
        ac = env.action_space.sample() # 没有使用，只为了解action的类型
        new = True #标记是一个episode的开始
        ob = env.reset()
        ob_adj = ob['adj']
        ob_node = ob['node']

        #当前episode的return
        cur_ep_ret = 0 
        cur_ep_ret_env = 0
        cur_ep_ret_d_step = 0
        cur_ep_ret_d_final = 0

        # 当前episode的长度
        cur_ep_len = 0
        cur_ep_len_valid = 0

        # 整个时间段中 完整 episodes的return
        ep_rets = []
        ep_rets_d_step = []
        ep_rets_d_final = []
        ep_rets_env = []

        # 整个时间段中 完整 episodes的长度
        ep_lens = []
        ep_lens_valid = []

        # episode最后一步的reward 为啥这么重要？
        ep_rew_final = []
        ep_rew_final_stat = []


        # 初始化时间序列
        ob_adjs = np.array([ob_adj for _ in range(horizon)])
        ob_nodes = np.array([ob_node for _ in range(horizon)])

        # 最终得到的ob 也就是生成的分子图
        ob_adjs_final = []
        ob_nodes_final = []

        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            self.pi.set_ac_real(self.ac_real)
            ac, vpred = self.pi.forward(ob['adj'],ob['node'])
            #print(ac)

            if t > 0 and t % horizon == 0:
                yield {"ob_adj": ob_adjs, "ob_node": ob_nodes, "ob_adj_final": np.array(ob_adjs_final), 
                "ob_node_final": np.array(ob_nodes_final), "rew": rews, "vpred": vpreds, "new": news,
                "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1-new), "ep_rets": ep_rets,
                "ep_lens": ep_lens, "ep_lens_valid": ep_lens_valid, "ep_final_rew": ep_rew_final,
                "ep_final_rew_stat": ep_rew_final_stat, "ep_rets_env": ep_rets_env,
                "ep_rets_d_step": ep_rets_d_step, "ep_rets_d_final": ep_rets_d_final}
                
                ep_rets = []
                ep_lens = []
                ep_lens_valid = []
                ep_rew_final = []
                ep_rew_final_stat = []
                ep_rets_d_step = []
                ep_rets_d_final = []
                ep_rets_env = []
                ob_adjs_final = []
                ob_nodes_final = []

            i = t % horizon
            ob_adjs[i] = ob["adj"]
            ob_nodes[i] = ob['node']
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew_env, new, info = env.step(ac)
            rew_d_step = 0
            if rew_env > 0:
                cur_ep_len_valid += 1
                
                rew_d_step = self.step_ratio * -1 * (loss_g_gen_discriminator(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :], self.dis_step)) / env.max_atom

            rew_d_final = 0
            if new:
                rew_d_final = self.final_ratio * -1 * (loss_g_gen_discriminator(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :], self.dis_final))

            rews[i] = rew_d_step + rew_d_final + rew_env

            cur_ep_ret += rews[i]
            cur_ep_ret_d_step += rew_d_step
            cur_ep_ret_d_final += rew_d_final
            cur_ep_ret_env += rew_env
            cur_ep_len += 1

            if new:
                with open("molecule_gen/"+name+'.csv', 'a') as f:
                    str = ''.join(['{},']*(len(info)+3))[:-1]+'\n'
                    f.write(str.format(info['smile'], info['reward_valid'], info['reward_qed'], info['reward_sa'], info['final_stat'], rew_env, rew_d_step, 
                    rew_d_final, cur_ep_ret, info['flag_steric_strain_filter'], info['flag_zinc_molecule_filter'], info['stop']))
                ob_adjs_final.append(ob['adj'])
                ob_nodes_final.append(ob['node'])
                ep_rets.append(cur_ep_ret_env)
                ep_rets_env.append(cur_ep_ret_env)
                ep_rets_d_step.append(cur_ep_ret_d_step)
                ep_rets_d_final.append(cur_ep_ret_d_final)
                ep_lens.append(cur_ep_len)
                ep_lens_valid.append(cur_ep_len_valid)
                ep_rew_final.append(rew_env)
                ep_rew_final_stat.append(info['final_stat'])
                cur_ep_ret = 0
                cur_ep_len = 0
                cur_ep_len_valid = 0
                cur_ep_ret_d_step = 0
                cur_ep_ret_d_final = 0
                cur_ep_ret_env = 0
                ob = env.reset()
            
            t += 1

# 用策略pi生成batch_size个完整的生成分子
def traj_final_generator(pi, env, batch_size):
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']
    ob_adjs = np.array([ob_adj for _ in range(batch_size)])
    ob_nodes = np.array([ob_node for _ in range(batch_size)])
    for i in range(batch_size):
        ob = env.reset()
        while True:
            ac, vpred = pi.forward(ob['adj'], ob['node'])
            ob, rew_env, new, info = env.step(ac)
            if new:
                ob_adjs[i] = ob['adj']
                ob_nodes[i] = ob['node']
                break
    return ob_adjs, ob_nodes


def add_vtarg_and_adv(seg, gamma, lam):
    '''
    计算At(adv), 与q函数(tdlamret)
    '''
    new = np.append(seg['new'],0)
    vpred = np.append(seg['vpred'], seg['nextvpred'].detach())
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg['rew']
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta= rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg['tdlamret'] = seg['adv'] + seg['vpred']

# mean(log(1-D(G(z)))) 生成器的loss
def loss_g_gen_discriminator(adj, node, dis):
    pred, logit = dis.forward(adj, node)
    loss_g_gen = -1 * torch.mean(F.binary_cross_entropy_with_logits(logit, torch.zeros(pred.shape)))
    return loss_g_gen

def learn(env, timesteps_per_actorbatch, gamma, lam, 
            optim_batchsize, optim_epochs, optim_lr, clip_param=0.2, entcoeff=0.01,
            expert_start=0, expert_end=1e6, rl_start=250, rl_end=1e6, curriculum_num=6, curriculum_step=200):
    pi = GCNPolicy()
    old_pi = GCNPolicy()
    dis_step = Discriminator()
    dis_final = Discriminator()

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)
    lenbuffer_valid = deque(maxlen=100)
    rewbuffer = deque(maxlen=100)
    rewbuffer_env = deque(maxlen=100)
    rewbuffer_d_step = deque(maxlen=100)
    rewbuffer_d_final = deque(maxlen=100)
    rewbuffer_final = deque(maxlen=100)
    rewbuffer_final_stat = deque(maxlen=100)

    traj_gen = traj_segment_generator(pi,env, timesteps_per_actorbatch, dis_step, dis_final)
    seg_gen = traj_gen.get_generator()
    counter = 0
    level = 0

    ## optim.Adam 把原来expert与ppo的参数两次更新拆开了
    adam_pi_expert = torch.optim.Adam(pi.parameters(), lr=optim_lr*0.05)
    adam_pi_ppo = torch.optim.Adam(pi.parameters(), lr=optim_lr*0.2)
    adam_step_dis = torch.optim.Adam(dis_step.parameters(), lr=optim_lr)
    adam_final_dis = torch.optim.Adam(dis_final.parameters(), lr=optim_lr)


    while True:
        seg = seg_gen.__next__()
        # 用新策略模型的参数为旧策略模型赋值，旧策略不需要计算梯度, 源码不在这，在ppo里 
        # 没理解为啥 本来就是从oldpi中采样才对，放在ppo的括号里每次pi都会等于old_pi, 没有体现重要性采样
        # 先这样试试
        for param_new, param_old in zip(pi.named_parameters(),old_pi.named_parameters()):
            assert param_new[0] == param_old[0]
            param_old[1].data = param_new[1].data
            param_old[1].requires_grad = False

        add_vtarg_and_adv(seg, gamma, lam)
        ob_adj, ob_node, ac, atarg, tdlamret = seg['ob_adj'], seg['ob_node'], seg['ac'], seg['adv'], seg['tdlamret']
        vpredbefore = seg['vpred']  # 更新前的预测value
        atarg = (atarg - atarg.mean()) / atarg.std() # 标准化atarg

        d = Dataset(dict(ob_adj=ob_adj, ob_node=ob_node, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=1)
        optim_batchsize = optim_batchsize or ob_adj.shape[0]

        for i_optim in range(optim_epochs):
            
            loss_expert = 0
            loss_expert_stop = 0
            g_expert = 0
            g_expert_stop = 0

            loss_d_step = 0
            loss_d_final = 0
            g_ppo = 0
            g_d_step = 0
            g_d_final = 0

            pretrain_shift = 5

            ## Expert
            if iters_so_far>=expert_start and iters_so_far<=expert_end+pretrain_shift:
                ob_expert, ac_expert = env.get_expert(optim_batchsize)
                pi.set_ac_real(ac_expert)
                ac_pred, v_pred = pi.forward(ob_expert['adj'],ob_expert['node'])

                # 这里有问题，原来使用ac_expert的 TODO
                pi_logp = pi.logp(ac_expert)

                loss_expert = - torch.mean(pi_logp) 

                adam_pi_expert.zero_grad()
                loss_expert.backward()
                adam_pi_expert.step()
            
            ## PPO
            if iters_so_far>=rl_start and iters_so_far<=rl_end:
                # 源码中把oldpi赋值放在这了 如果不行的话可能真要放这 TODO
                batch = d.next_batch(optim_batchsize)
                if iters_so_far >= rl_start+pretrain_shift:  # 等判别器训练一段时间,后再启动traj_generator,因为需要判别器的奖励
                    pi.set_ac_real(batch["ac"])
                    old_pi.set_ac_real(batch["ac"])
                    batch_acs = batch["ac"]
                    atarg = batch["atarg"]
                    # 又想了一哈，源码中batch['ob']其实是不需要的，batch['ob']是已经经过pi的ob，但目前要跑一次算loss
                    pi.forward(batch["ob_adj"],batch["ob_node"])
                    old_pi.forward(batch["ob_adj"],batch["ob_node"])
                    pi_logp = pi.logp(batch_acs)
                    old_pi_logp = old_pi.logp(batch_acs)

                    # 熵惩罚，不能让熵太小，要让agent可以有更多选择
                    ent = pi.entorpy()
                    meanent = torch.mean(ent)
                    pol_entpen = (-entcoeff) * meanent
                    # 两个分布的kl散度
                    kl_oldnew = old_pi.kl(pi.pd)
                    meankl = torch.mean(kl_oldnew)

                    ratio = torch.exp(pi_logp - old_pi_logp) # pnew(ac)/pold(ac)
                    print(type(ratio))
                    atarg = torch.Tensor(atarg)
                    surr1 = ratio * atarg
                    surr2 = torch.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
                    loss_surr = - torch.mean(torch.minimum(surr1, surr2)) # 加负号将目标函数值变为loss
                    
                    ret = torch.Tensor(batch["vtarg"])
                    vf_loss = torch.mean(torch.square(pi.vpred - ret)) # 价值函数的loss
                    total_loss = loss_surr + pol_entpen + vf_loss  # PPO阶段的loss
                    losses = {"loss_surr":loss_surr, "pol_entpen":pol_entpen, "vf_loss":vf_loss, "kl":meankl, "entropy":meanent}

                    adam_pi_ppo.zero_grad()
                    total_loss.backward()
                    adam_pi_ppo.step()
                if i_optim >= optim_epochs//2:
                    # 更新过程判别器
                    ob_expert ,_ = env.get_expert(optim_batchsize)
                    step_pred_real, step_logit_real = dis_step.forward(ob_expert["adj"], ob_expert["node"])
                    step_pred_gen, step_logit_gen = dis_step.forward(batch["ob_adj"], batch["ob_node"])

                    loss_d_step_real = torch.mean(F.binary_cross_entropy_with_logits(step_logit_real, torch.ones(step_pred_real.shape)*0.9))
                    loss_d_step_gen = torch.mean(F.binary_cross_entropy_with_logits(step_logit_gen, torch.zeros(step_pred_gen.shape)))
                    loss_d_step = loss_d_step_real + loss_d_step_gen

                    adam_step_dis.zero_grad()
                    loss_d_step.backward()
                    adam_step_dis.step()

                if i_optim >= optim_epochs//4*3:
                    # 更新结果判别器
                    ob_expert, _ = env.get_expert(optim_batchsize)
                    seg_final_adj, seg_final_node = traj_final_generator(pi, copy.deepcopy(env), optim_batchsize)
                    final_pred_real, final_logit_real = dis_final.forward(ob_expert['adj'], ob_expert['node'])
                    final_pred_gen, final_logit_gen = dis_final.forward(seg_final_adj, seg_final_node)
                    loss_d_final_real = torch.mean(F.binary_cross_entropy_with_logits(final_logit_real, torch.ones(final_pred_real.shape)*0.9))
                    loss_d_final_gen = torch.mean(F.binary_cross_entropy_with_logits(final_logit_gen, torch.zeros(final_pred_gen.shape)))
                    loss_d_final = loss_d_final_real + loss_d_final_gen

                    adam_final_dis.zero_grad()
                    loss_d_final.backward()
                    adam_final_dis.step()

        iters_so_far += 1
        counter += 1
        if (not counter % curriculum_step) and counter//curriculum_step < curriculum_num:
            level += 1

        with open('molecule_gen/test.csv', 'a') as f:
                f.write('***** Iteration {} *****\n'.format(iters_so_far))

def main():
    print("i")
    print(os.path.abspath("molecule_gen"))
    if not os.path.exists("molecule_gen"):
        print("makedirs")
        os.makedirs("./molecule_gen")
    env = gym.make("molecule-v0")
    env.init(reward_type= "qedsa")
    print(env.observation_space)
    # 256 32 8
    learn(env, 256, 1, 0.95, 32, 8, 1e-3)

if __name__ == '__main__':
    main()