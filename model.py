from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchtext
import torchvision
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
from util.utils import *
from util.C_GCN import C_GCN
from util.Position import PositionEncoder
from util.Summarization import Summarization
from util.AGSA import AGSA


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False, head=16, smry_k=12, drop=0.0):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm, head, drop, smry_k)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


# grid features
class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False, head=16, drop=0.0, smry_k=12):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        # context enhanced grid features
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        self.mvs = Summarization(embed_size, smry_k)
        # MLP
        hidden_size = embed_size
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(drop)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        r0 = np.sqrt(6.) / np.sqrt(self.fc1.in_features + self.fc1.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-r0, r0)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-r0, r0)
        self.fc2.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        features_mean = torch.mean(features, 1)
        '''choose whether to l2norm'''
        if not self.no_imgnorm:
            features_mean = l2norm(features_mean)

        return features, features_mean

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# region features
class EncoderImagePrecompSelfAttn(nn.Module):

    def __init__(self, img_dim, embed_size, head, smry_k, drop=0.0, no_imgnorm=False):
        super(EncoderImagePrecompSelfAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        if not self.no_imgnorm:
            fc_img_emd = l2norm(fc_img_emd, dim=-1)

        features_mean = torch.mean(fc_img_emd, 1)
        if not self.no_imgnorm:
            features_mean = l2norm(features_mean)

        return fc_img_emd, features_mean

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompSelfAttn, self).load_state_dict(new_state)


''' Text encoder'''

class EncoderText(nn.Module):
    '''This func can utilize w2v initialization for word embedding'''

    def __init__(self, wemb_type, word2idx, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bidirectional_RNN=True, no_txtnorm=False,
                 use_abs=False, RNN_type='GRU', head=12, drop=0.0):

        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.use_bidirectional_RNN = use_bidirectional_RNN
        self.RNN_type = RNN_type
        if RNN_type == 'GRU':
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)
        elif RNN_type == 'LSTM':
            self.rnn = nn.LSTM(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)

        self.dropout = nn.Dropout(opt.dropout_rate)

        '''change here'''
        self.init_weights(wemb_type, word2idx, word_dim)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_RNN:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # take absolute value, used by order embeddings
        if self.use_abs:
            cap_emb = torch.abs(cap_emb)

        cap_emb_mean = torch.mean(cap_emb, 1)

        if not self.no_txtnorm:
            cap_emb_mean = l2norm(cap_emb_mean)

        return cap_emb, cap_emb_mean


def sequence_mask(lengths, max_len=None, inverse=False):
    ''' Creates a boolean mask from sequence lengths.
    '''
    # lengths: LongTensor, (batch, )
    batch_size = lengths.size(0)
    max_len = max_len or lengths.max()
    mask = torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1)
    if inverse:
        mask = mask.ge(lengths.unsqueeze(1))
    else:
        mask = mask.lt(lengths.unsqueeze(1))
    return mask


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = Block(dim, num_clusters)
        self.centroids = nn.Parameter(0.1 * torch.rand(num_clusters, dim))
        self._init_params()
        self.linear = nn.Linear(self.num_clusters * self.dim, 1024)

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x_, length, input_modal):
        bs, num, emb_dim = x_.size()
        if input_modal == 'textual' or 'region':
            x = x_.unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
        else:
            x = x_.unsqueeze(-1).permute(0, 2, 1, 3).view(bs, emb_dim, self.num_clusters,
                                                          -1).contiguous()  # [N, M, dim, 1] -> [N, dim, M, 1]

        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)

        # mask
        if input_modal == 'textual':
            mask = sequence_mask(length, x.size(2), inverse=False).unsqueeze(1).cuda()
            mask = mask.expand_as(soft_assign)
            pad_masks = (mask == 0)
            soft_assign = soft_assign.masked_fill(pad_masks, -1e18)

        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - self.centroids.expand(
            x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)  #
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        # vlad = self.linear(vlad)
        vlad = vlad.view(-1, self.num_clusters * self.dim).contiguous()

        return vlad


''' Visual self-attention module '''
class V_single_modal_atten(nn.Module):
    """
    Single Visual Modal Attention Network.
    """

    def __init__(self, image_dim, embed_dim, cluster_dim, smry_k, use_bn, activation_type, dropout_rate, img_region_num,
                 cluster_num):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(V_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space

        self.fc2 = nn.Linear(image_dim, embed_dim)  # embed memory to common space
        self.fc2_2 = nn.Linear(image_dim, embed_dim)

        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        self.fc4 = nn.Linear(image_dim, embed_dim)  # embed attentive feature to common space

        if use_bn == True and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.BatchNorm1d(img_region_num + 36),
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == False and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
        elif use_bn == True and activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.BatchNorm1d(img_region_num + 36),
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        else:
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, v_t, m_v):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_v = self.embedding_1(v_t)

        if m_v.size()[-1] == v_t.size()[-1]:
            W_v_m = self.embedding_2(m_v)
        else:
            W_v_m = self.embedding_2_2(m_v)

        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_v.size()[1], 1)

        h_v = W_v.mul(W_v_m)

        a_v = self.embedding_3(h_v)
        a_v = a_v.squeeze(2)
        weights = self.softmax(a_v)

        v_att = ((weights.unsqueeze(2) * v_t)).sum(dim=1)

        # l2 norm
        v_att = l2norm((v_att))

        return v_att, weights


''' Textual self-attention module '''


class T_single_modal_atten(nn.Module):
    """
    Single Textual Modal Attention Network.
    """

    def __init__(self, embed_dim, use_bn, activation_type, dropout_rate):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(T_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = l2norm(u_att)

        return u_att, weights


class fusion(nn.Module):
    def __init__(self, embed_size, hidden_size, head, drop, smry_k):
        super(fusion, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.mvs = Summarization(embed_size, smry_k)
        # context-enhanced embedding
        self.agsa = AGSA(2, self.embed_size, h=head, is_share=False, drop=drop)
        # MLP
        self.fc1 = nn.Linear(self.embed_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.embed_size)
        self.bn = nn.BatchNorm1d(self.embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, features):
        bs, region_num = features.size()[:2]
        # gated fusion
        agsa_emb = self.agsa(features)
        x = self.fc2(self.dropout(F.relu((self.fc1(agsa_emb)))))
        x = (self.bn(x.view(bs * region_num, -1))).view(bs, region_num, -1)
        x = agsa_emb + self.dropout(x)
        x = l2norm(x)

        # multi-scale fusion
        smry_mat = self.mvs(x)
        L = F.softmax(smry_mat, dim=1)
        img_emb_mat = torch.matmul(L.transpose(1, 2), x)  # (bs, k, dim)
        multi_level_emb = F.normalize(img_emb_mat, dim=-1)

        return multi_level_emb


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
    def forward(self, im_g, s_g):
        # compute image-sentence score matrix
        scores = self.sim(im_g, s_g)
        diagonal = scores.diag().view(im_g.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class CVSE(object):
    """
    CVSE model
    """

    def __init__(self, word2idx, opt):

        self.grad_clip = opt.grad_clip
        self.dataset_name = opt.data_name
        self.GT_label_ratio = opt.Concept_label_ratio
        self.head = 64
        self.smry_k = 24
        # self.img_enc = EncoderImage(opt)
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm, head=self.head, smry_k=self.smry_k, drop=0.5)

        self.img_enc1 = EncoderImagePrecompSelfAttn(opt.img_dim, opt.embed_size, self.head, self.smry_k, 0.5, no_imgnorm=opt.no_imgnorm)

        self.txt_enc = EncoderText(opt.wemb_type, word2idx, opt,
                                   opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bidirectional_RNN=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm,
                                   use_abs=opt.use_abs, head=self.head, drop=0.5)
        self.fusing = fusion(opt.embed_size, opt.embed_size, self.head, drop=0.5, smry_k=self.smry_k)

        self.dim = 1024
        self.cluster_num = 12

        # separate netvlad module
        self.net_vlad = NetVLAD(num_clusters=self.cluster_num, dim=self.dim, alpha=1.0)

        img_region_num = 49
        self.fuse_weight = 0.85

        # visual self-attention
        self.V_self_atten_enhance = V_single_modal_atten(opt.img_dim, opt.embed_size, self.cluster_num, self.smry_k,
                                                         opt.use_BatchNorm,
                                                         opt.activation_type, opt.dropout_rate, img_region_num,
                                                         self.cluster_num)
        # textual self-attention
        self.T_self_atten_enhance = T_single_modal_atten(opt.embed_size, opt.use_BatchNorm,
                                                         opt.activation_type, opt.dropout_rate)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.V_self_atten_enhance.cuda()
            self.T_self_atten_enhance.cuda()
            self.V_self_atten_enhance.cuda()
            self.T_self_atten_enhance.cuda()
            self.net_vlad.cuda()
            self.img_enc1.cuda()
            self.fusing.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        ### 1. loss
        self.criterion_rank = ContrastiveLoss(margin=opt.margin,
                                              measure=opt.measure,
                                              max_violation=opt.max_violation)

        ### 2. learnable parms
        params = self.get_config_optim(opt.learning_rate, opt.learning_rate_MLGCN)

        ## 3. optimizer
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        params = list(self.img_enc.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.V_self_atten_enhance.parameters())
        params += list(self.T_self_atten_enhance.parameters())
        params += list(self.net_vlad.parameters())
        params += list(self.img_enc1.parameters())
        params += list(self.fusing.parameters())
        self.params = params

        self.Eiters = 0

    def get_config_optim(self, lr_base, lr_MLGCN):
        return [
            {'params': self.img_enc.parameters(), 'lr': lr_base},
            {'params': self.txt_enc.parameters(), 'lr': lr_base},
            {'params': self.V_self_atten_enhance.parameters(), 'lr': lr_base},
            {'params': self.T_self_atten_enhance.parameters(), 'lr': lr_base},
            {'params': self.net_vlad.parameters(), 'lr': lr_base},
            {'params': self.img_enc1.parameters(), 'lr': lr_base},
            {'params': self.fusing.parameters(), 'lr': lr_base},
        ]

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.V_self_atten_enhance.state_dict(),
                      self.T_self_atten_enhance.state_dict(),
                      self.net_vlad.state_dict(),
                      self.img_enc1.state_dict(),
                      self.fusing.state_dict(),
                      ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.V_self_atten_enhance.load_state_dict(state_dict[2])
        self.T_self_atten_enhance.load_state_dict(state_dict[3])
        self.net_vlad.load_state_dict(state_dict[4])
        self.img_enc1.load_state_dict(state_dict[5])
        self.fusing.load_state_dict(state_dict[6])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.V_self_atten_enhance.train()
        self.T_self_atten_enhance.train()
        self.net_vlad.train()
        self.img_enc1.train()
        self.fusing.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.V_self_atten_enhance.eval()
        self.T_self_atten_enhance.eval()
        self.net_vlad.eval()
        self.img_enc1.eval()
        self.fusing.eval()

    def forward_emb(self, images, images_region, captions, lengths, alpha, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        images_region = Variable(images_region, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            images_region = images_region.cuda()
            captions = captions.cuda()

        img_emb, img_emb_mean = self.img_enc(images)
        obj_emb, obj_emb_mean = self.img_enc1(images_region)
        cap_emb, cap_emb_mean = self.txt_enc(captions, lengths)
        lengths = torch.tensor(lengths).cuda()

        emb_v = torch.cat((img_emb, obj_emb), 1)
        emv_v_mean = torch.cat((img_emb_mean, obj_emb_mean), 1)
        fine_grained_emb = self.fusing(emb_v)

        # shared NetVlad module
        vlad_v_emb = self.net_vlad(fine_grained_emb, lengths, input_modal=None)

        vlad_t_emb = self.net_vlad(cap_emb, lengths, input_modal='textual')

        # global features
        instance_emb_v, visual_weights = self.V_self_atten_enhance(emb_v, emv_v_mean)
        instance_emb_t, textual_weights = self.T_self_atten_enhance(cap_emb, cap_emb_mean)
        v_emb = torch.cat((instance_emb_v, vlad_v_emb), 1)
        t_emb = torch.cat((instance_emb_t, vlad_t_emb), 1)

        return v_emb, t_emb

    def forward_loss(self, v_emb, t_emb, dataset_name, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss_rank_2_stream = self.criterion_rank(v_emb, t_emb)

        loss = loss_rank_2_stream

        self.logger.update('Le_rank_fused', loss_rank_2_stream.item(), v_emb.size(0))

        return loss

    def train_emb(self, images, images_region, captions, lengths, ids=None, *args):

        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('GCN_lr', self.optimizer.param_groups[4]['lr'])

        # compute the embeddings
        '''! change for adding input w2v dict for GCN attribute predictor'''
        v_emb, t_emb = self.forward_emb(images, images_region, captions, lengths, self.fuse_weight)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(v_emb, t_emb, self.dataset_name)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
