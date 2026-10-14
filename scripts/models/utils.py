import torch
import numpy as np
import torch.nn as nn

import warnings

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))

class HumanHumanAttention(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, config_master, config_model):
        super(HumanHumanAttention, self).__init__()
        self.config_master = config_master
        self.config_model = config_model

        self.num_attn_heads = config_model.HHAttn_num_heads
        self.attn_size = config_model.HHAttn_attn_size

        self.q_linear = init_(nn.Linear(self.attn_size, self.attn_size))
        self.v_linear = init_(nn.Linear(self.attn_size, self.attn_size))
        self.k_linear = init_(nn.Linear(self.attn_size, self.attn_size))

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads, batch_first=True)


    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, detected_humans, batch_size, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        
        mask = torch.zeros(batch_size, max_human_num + 1).to(self.config_master.config_general.device)
        mask[torch.arange(batch_size), detected_humans.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1]
        return mask


    def forward(self, inp, detected_humans):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        batch_size, max_human_num, _ = inp.size()
        attn_mask = self.create_attn_mask(detected_humans, batch_size, max_human_num)  # [seq_len*nenv, 1, max_human_num]

        # input_emb=self.embedding_layer(inp)
        # q=self.q_linear(input_emb)
        # k=self.k_linear(input_emb)
        # v=self.v_linear(input_emb)

        # already embedded
        q=self.q_linear(inp)
        k=self.k_linear(inp)
        v=self.v_linear(inp)

        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        return z



class RobotHumanAttention(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, config_master, config_model):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(RobotHumanAttention, self).__init__()

        self.config_master = config_master
        self.config_model = config_model
        self.device = config_master.config_general.device

        self.robot_embedding_size = config_model.robot_embedding_size
        self.human_embedding_size = config_model.HHAttn_attn_size
        self.attn_size = config_model.final_attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.num_attention_head = 2
        self.robot_query_embedding = nn.ModuleList()
        self.HH_key_embedding =nn.ModuleList()
        self.HH_value_embedding = nn.ModuleList()

        for i in range(self.num_attention_head):
            self.robot_query_embedding.append(nn.Sequential(init_(nn.Linear(self.robot_embedding_size, self.attn_size)),nn.LayerNorm(self.attn_size), nn.ReLU(),
                                             init_(nn.Linear(self.attn_size, self.attn_size)), nn.LayerNorm(self.attn_size)))

            # Linear layer to embed spatial edgeRNN hidden states
            self.HH_key_embedding.append(nn.Sequential(init_(nn.Linear(self.human_embedding_size, self.attn_size)),nn.LayerNorm(self.attn_size), nn.ReLU(),
                                             init_(nn.Linear(self.attn_size, self.attn_size)), nn.LayerNorm(self.attn_size)))
            
            self.HH_value_embedding.append(nn.Sequential(init_(nn.Linear(self.human_embedding_size, self.attn_size)),nn.LayerNorm(self.attn_size), nn.ReLU(),
                                             init_(nn.Linear(self.attn_size, self.attn_size)), nn.LayerNorm(self.attn_size)))
        
        if self.num_attention_head > 1:
            self.final_attn_linear = init_(nn.Linear(self.num_attention_head*self.attn_size, self.attn_size))

    def create_attn_mask(self, detected_humans, batch_size, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        mask = torch.zeros(batch_size, max_human_num + 1).to(self.device)
        mask[torch.arange(batch_size), detected_humans.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1]  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, Q, K, V, attn_mask=None):
        batch_size, max_humans, h_size = V.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        
        Q = torch.unsqueeze(Q, dim=1)
        attn = Q * K
        attn = torch.sum(attn, dim=2)

        # Variable length
        temperature = 1 / np.sqrt(self.attn_size)
        attn = attn * temperature

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        V = V.permute(0,2,1)
        # h_spatials = h_spatials.view(batch_size, self.agent_num, self.human_num, h_size)
        # h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                        #  1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(V, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1)
        return weighted_value, attn


    def forward(self, robot_embedding, HH_embedding, detected_humans):
        batch_size, max_human_num, _ = HH_embedding.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.max_human_num = max_human_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            robot_embedding_query = self.robot_query_embedding[i](robot_embedding)

            # Embed the spatial edgeRNN hidden states
            HH_embedding_keys = self.HH_key_embedding[i](HH_embedding)
            HH_embedding_values = self.HH_value_embedding[i](HH_embedding)

            # Dot based attention
            # robot_embedding_query = robot_embedding_query.repeat_interleave(self.max_human_num, dim=2)

            
            attn_mask = self.create_attn_mask(detected_humans, batch_size, max_human_num)
            
            weighted_value,attn=self.att_func(robot_embedding_query, HH_embedding_keys, HH_embedding_values, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]
        

class HumanAvoidanceNetwork(nn.Module):

    def __init__(self, config_master, config_model):

        super(HumanAvoidanceNetwork, self).__init__()

        self.config_master = config_master
        self.config_model = config_model
        self.human_num = config_master.config_HA.sim.max_allowable_humans #config_master.config_HA.sim.human_num + config_master.config_HA.humans.num_static_humans

        robot_embedding_size = config_model.robot_embedding_size
        human_embedding_size = config_model.human_embedding_size

        # robot embedding
        self.robot_size = config_model.lookback * 2
       
        self.robot_linear = nn.Sequential(nn.Linear(self.robot_size, robot_embedding_size), nn.LayerNorm(robot_embedding_size), nn.ReLU(),
                                          nn.Linear(robot_embedding_size, robot_embedding_size), nn.LayerNorm(robot_embedding_size), nn.ReLU(),
                                            nn.Linear(robot_embedding_size, robot_embedding_size), nn.LayerNorm(robot_embedding_size)) 


        # HH attention
        self.lookback = config_master.config_general.env.lookback
        self.human_input_size = (self.lookback+1) * 2
        self.HH_attn_size = config_model.HHAttn_attn_size
                    
        self.human_gru_encoder = nn.GRU(2, self.HH_attn_size, batch_first=True)
        self.human_gru_encoder.flatten_parameters()
        
        self.spatial_attn = HumanHumanAttention(config_master, config_model)
        self.spatial_linear = nn.Sequential(init_(nn.Linear(self.spatial_attn.attn_size, self.spatial_attn.attn_size)), nn.LayerNorm(self.spatial_attn.attn_size), nn.ReLU(),
                                            init_(nn.Linear(self.spatial_attn.attn_size, self.spatial_attn.attn_size)), nn.LayerNorm(self.spatial_attn.attn_size), nn.ReLU(),
                                            init_(nn.Linear(self.spatial_attn.attn_size, self.spatial_attn.attn_size)),  nn.LayerNorm(self.spatial_attn.attn_size), nn.ReLU(),
                                            init_(nn.Linear(self.spatial_attn.attn_size, self.spatial_attn.attn_size)), nn.LayerNorm(self.spatial_attn.attn_size), nn.ReLU(),
                                            init_(nn.Linear(self.spatial_attn.attn_size, self.spatial_attn.attn_size)))

        
        # robot self attention on HH multi head output
        self.attn = RobotHumanAttention(config_master, config_model)


        # robot self attention on HH multi head output
        self.HHattenEmbedding = nn.Sequential(init_(nn.Linear(config_model.final_attention_size, human_embedding_size)), nn.ReLU(),
                                              init_(nn.Linear(human_embedding_size, human_embedding_size)), nn.LayerNorm(human_embedding_size), nn.ReLU(),
                                              init_(nn.Linear(human_embedding_size, human_embedding_size)), nn.LayerNorm(human_embedding_size), nn.ReLU(),
                                              init_(nn.Linear(human_embedding_size, human_embedding_size)), nn.LayerNorm(human_embedding_size), nn.ReLU(),
                                              init_(nn.Linear(human_embedding_size, human_embedding_size)), nn.LayerNorm(human_embedding_size), nn.ReLU(),
                                              init_(nn.Linear(human_embedding_size, human_embedding_size)))
        
        #  MLP Embedding for robot_embedding + HH_attn_output
        self.robot_further_embedding = nn.Sequential(init_(nn.Linear(robot_embedding_size, robot_embedding_size)), nn.LayerNorm(robot_embedding_size), nn.ReLU(),
                                                     init_(nn.Linear(robot_embedding_size, robot_embedding_size)), nn.ReLU())
        
        input_size_for_combineMLP = robot_embedding_size + human_embedding_size
        # combine the two embeddings and create a hidden state
        # the result is the shared embedding space for the actor critic
        shared_latent_space_dim = config_model.shared_latent_space
        self.combineMLP = nn.Sequential(init_(nn.Linear(input_size_for_combineMLP, shared_latent_space_dim)), nn.LayerNorm(shared_latent_space_dim), nn.ReLU(),
                                        init_(nn.Linear(shared_latent_space_dim, shared_latent_space_dim)), nn.LayerNorm(shared_latent_space_dim), nn.ReLU(),
                                        init_(nn.Linear(shared_latent_space_dim, shared_latent_space_dim)), nn.LayerNorm(shared_latent_space_dim), nn.ReLU(),
                                        init_(nn.Linear(shared_latent_space_dim, shared_latent_space_dim)), nn.LayerNorm(shared_latent_space_dim), nn.ReLU(),
                                        init_(nn.Linear(shared_latent_space_dim, shared_latent_space_dim)), nn.LayerNorm(shared_latent_space_dim), nn.ReLU())

       

    def forward(self, inputs_HA, inputs_PT):

        batch_size = inputs_HA['robot_node'].shape[0]

        # robot_node = reshapeT(inputs_DO['robot_node'], seq_length, nenv)
        # temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        # spatial_edges = reshapeT(inputs_DO['spatial_edges'], seq_length, nenv)

        robot_node = inputs_HA['robot_node']
        spatial_edges = inputs_HA['spatial_edges']
        detected_human_num = inputs_HA['detected_human_num'].cpu().int()

        robot_states = self.robot_linear(robot_node)

        # attention modules 
        # human-human attention
           
        flattened_spatial_edges = spatial_edges.view(-1, self.lookback+1, 2) # [Bxmax_num_humans, lookback+1, 2]
        
        gru_output = self.human_gru_encoder(flattened_spatial_edges)[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gru_output = self.human_gru_encoder(flattened_spatial_edges)[0] 
            if len(w) > 0:
                print('Warning in gru_output: ', w)
                self.human_gru_encoder.flatten_parameters()

        valid_indices = inputs_HA['human_valid_history'].long() - 1
        valid_indices = valid_indices.view(flattened_spatial_edges.shape[0])
        whole_numbers_counter = torch.arange(flattened_spatial_edges.shape[0])
        relevant_embeddings = gru_output[whole_numbers_counter, valid_indices, :]
        spatial_edges_embedded = relevant_embeddings.view(batch_size, self.human_num, -1) 

        spatial_attn_out=self.spatial_attn(spatial_edges_embedded, detected_human_num)
        
        output_spatial = self.spatial_linear(spatial_attn_out)

        # robot-human attention
        hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        
        # outputs = self.humanNodeMLP(torch.cat((outputs, hidden_attn_weighted), dim=-1))
        HH_attn_embedding = self.HHattenEmbedding(hidden_attn_weighted)
        robot_embedding = self.robot_further_embedding(robot_states)

        combined_embedding_input = torch.cat((HH_attn_embedding, robot_embedding), dim=-1)
        combined_embedding = self.combineMLP(combined_embedding_input)

        # x is the output and will be sent to actor and critic
        x = combined_embedding
        if self.config_model.use_time:
            percent_episode = inputs_HA['percent_episode']
            x = torch.cat((x, percent_episode), dim=-1)
            
        return x

def create_MLP(input_size, hidden_layers, output_size, last_activation=True, use_batch_norm=False, dropout_rate=None, use_layer_norm=False):
    layers = []
    if use_batch_norm and use_layer_norm:
        raise ValueError('Cannot use both batch and layer normalization')
    
    for i in range(len(hidden_layers)):
        if i == 0:
            layers.append(init_(nn.Linear(input_size, hidden_layers[i])))
        else:
            layers.append(init_(nn.Linear(hidden_layers[i-1], hidden_layers[i])))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
        elif use_layer_norm:  # Add layer normalization if specified
            layers.append(nn.LayerNorm(hidden_layers[i]))

        layers.append(nn.ReLU())

        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

    layers.append(init_(nn.Linear(hidden_layers[-1], output_size)))
    if last_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ActorLinear(nn.Module):
    def __init__(self, config_master, config_model, additional_input=0):
        super(ActorLinear, self).__init__()
        self.config_master = config_master
        self.config_model = config_model
       
        self.actor = create_MLP(config_model.size_of_fused_layers + additional_input, config_model.actor_layers, 2 * config_master.config_general.action_dim, last_activation=False, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)

    def forward(self, x):
        dist_params = self.actor(x)
        return dist_params
    

class CriticLinear(nn.Module):
    def __init__(self, config_master, config_model, additional_input=0, add_action=True):
        super(CriticLinear, self).__init__()
        
        self.config_master = config_master
        self.config_model = config_model
        self.add_action = add_action

        if self.add_action:
            input_size = config_model.size_of_fused_layers + additional_input + 2
        else:
            input_size = config_model.size_of_fused_layers + additional_input

        self.critic = create_MLP(input_size, config_model.critic_layers, 1, last_activation=False, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)

    def forward(self, x, action=None):
        if self.add_action:
            value = self.critic(torch.cat((x, action), dim=-1))
        else:
            value = self.critic(x)
        
        return value