from ...torch_core import *
from ...layers import *

__all__ = ['EmbeddingDropout', 'LinearDecoder', 'MultiBatchRNNCore', 'PoolingLinearClassifier', 'RNNCore', 'RNNDropout', 
           'SequentialRNN', 'WeightDropout', 'dropout_mask', 'get_language_model', 'get_rnn_classifier', 'get_seq2seq']

def dropout_mask(x:Tensor, sz:Collection[int], p:float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

def get_shape(inp):
    ret = []

    if type(inp) in [tuple, list]:
        ret.append(len(inp))
        ret.append(get_shape(inp[0]))
    elif type(inp) is Tensor:
        ret.append(inp.shape)
    else:
        ret.append(1)

    return ret

class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p:float=0.5):
        super().__init__()
        self.p=p

    def forward(self, x:Tensor)->Tensor:
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

class WeightDropout(nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module:nn.Module, weight_p:float, layer_names:Collection[str]=['weight_hh_l0']):
        super().__init__()
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args:ArgStar):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()

class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, emb:nn.Module, embed_p:float):
        super().__init__()
        self.emb,self.embed_p = emb,embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words:LongTensor, scale:Optional[float]=None)->Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)

class RNNCore(nn.Module):
    "AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False):

        super().__init__()
        self.bs,self.qrnn,self.ndir = 1, qrnn,(2 if bidir else 1)
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires an installation of cuda
            from .qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True,
                                   use_cuda=torch.cuda.is_available()) for l in range(n_layers)]
            for rnn in self.rnns: rnn.linear = WeightDropout(rnn.linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                1, bidirectional=bidir, batch_first=True) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        input, enc_batch_extend_vocab, extra_zeros, _, dec_padding_mask, dec_lens, output = input
        bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        hidden = new_hidden.copy()
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs, hidden, output, enc_batch_extend_vocab, extra_zeros, dec_padding_mask, dec_lens

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

class LinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_out:int, n_hid:int, output_p:float, tie_encoder:nn.Module=None, bias:bool=True):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output)
        return decoded, raw_outputs, outputs

class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

class MultiBatchRNNCore(RNNCore):
    "Create a RNNCore module that can process a full sentence."

    def __init__(self, bptt:int, max_seq:int, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs,sl = input.size()
        self.reset()
        raw_outputs, outputs = [],[]
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[:,i: min(i+self.bptt, sl)])
            if i>(sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)

class PoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.transpose(1,2), (1,)).view(bs,-1)

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        bs,sl,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[:,-1], mxpool, avgpool], 1)
        x = self.layers(x)
        return x, raw_outputs, outputs

def get_language_model(vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, tie_weights:bool=True,
                       qrnn:bool=False, bias:bool=True, bidir:bool=False, output_p:float=0.4, hidden_p:float=0.2, input_p:float=0.6,
                       embed_p:float=0.1, weight_p:float=0.5)->nn.Module:
    "Create a full AWD-LSTM."
    rnn_enc = RNNCore(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, qrnn=qrnn, bidir=bidir,
                 hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = rnn_enc.encoder if tie_weights else None
    model = SequentialRNN(rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias))
    model.reset()
    return model

def get_rnn_classifier(bptt:int, max_seq:int, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int,
                       pad_token:int, layers:Collection[int], drops:Collection[float], bidir:bool=False, qrnn:bool=False,
                       hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5)->nn.Module:
    "Create a RNN classifier model."
    rnn_enc = MultiBatchRNNCore(bptt, max_seq, vocab_sz, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      qrnn=qrnn, hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    model = SequentialRNN(rnn_enc, PoolingLinearClassifier(layers, drops))
    model.reset()
    return model

# ---********************** seq2seq **********************--- #

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, hidden[0].shape[0], 1, -1)
#         embedded = self.dropout(embedded)
#
#         print(embedded[0].shape, hidden[0].shape)
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=2)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=defaults.device)
#
# class MultiBatchRNNCoreDecoder(AttnDecoderRNN):
#     "Create a RNNCore module that can process a full sentence."
#
#     def __init__(self, bptt:int, max_seq:int, *args, **kwargs):
#         self.max_seq,self.bptt = max_seq,bptt
#         super().__init__(*args, **kwargs)
#
#     def concat(self, arrs:Collection[Tensor])->Tensor:
#         "Concatenate the `arrs` along the batch dimension."
#         return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
#
#     def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
#         encoder_outputs, encoder_hidden = input
#         sos_token = 0
#         decoder_input = torch.tensor([sos_token]*encoder_hidden[0].shape[0], device=defaults.device)
#         decoder_hidden = encoder_hidden
#         raw_outputs, outputs = [], []
#         for di in range(self.max_seq):
#             decoder_output, decoder_hidden, decoder_attention = super().forward(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             raw_outputs.append(decoder_output)
#             outputs.append(decoder_hidden)
#         return self.concat(raw_outputs), encoder_outputs, encoder_hidden

class BaseRNN(nn.Module):
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn, combined

class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, emb_sz, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, teacher_forcing_ratio=0.):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.bidirectional_encoder = bidirectional
        self.ndir = 2 if bidirectional else 1
        # self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.rnns = [self.rnn_cell(emb_sz if l == 0 else hidden_size, (hidden_size if l != n_layers - 1 else emb_sz) // self.ndir,
                             1, bidirectional=bidirectional, batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList(self.rnns)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding = nn.Embedding(self.output_size, emb_sz)
        if use_attention:
            self.attention = Attention(emb_sz)

        self.out = nn.Linear(emb_sz, self.output_size)
        self.emb_sz = emb_sz

        self.pointer_gen = True
        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(3 * emb_sz, 1)

    def forward_step(self, input_var, hidden, encoder_outputs, enc_batch_extend_vocab, extra_zeros, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        # print(input_var)
        # print(self.output_size)
        # torch.clamp_(input_var,min=0,max=self.output_size-1)
        input_var[input_var >= self.output_size] = 0
        # print(input_var)
        # print(self.output_size)
        embedded = self.embedding(input_var)
        raw_output = self.input_dropout(embedded)

        for l, rnn in enumerate(self.rnns):
            raw_output, hidden[l] = rnn(raw_output, hidden[l])
            if l == self.n_layers - 1: output = raw_output

        attn = None
        if self.use_attention:
            output, attn, c_t = self.attention(output, encoder_outputs)

        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, embedded), 2)  # B x seq_len x (3*emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)  # B x seq_len x 1
            p_gen = torch.sigmoid(p_gen)

        vocab_dist = F.softmax(self.out(output.contiguous().view(-1, self.emb_sz)), dim=1).view(batch_size,
                                                                                               output_size, -1)

        # print('p_gen::shape', get_shape(p_gen))
        # print('vocab_dist::shape', get_shape(vocab_dist))

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist  # B x seq_len x output_size
            attn_dist_ = (1 - p_gen) * attn  # B x seq_len x input_size

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros[:,:vocab_dist_.shape[1],:]], 2)

            enc_batch_extend_vocab = enc_batch_extend_vocab.unsqueeze(1).expand(-1,attn_dist_.shape[1],-1)

            final_dist = vocab_dist_.scatter_add(2, enc_batch_extend_vocab, attn_dist_)
            # final_dist = function(final_dist, dim=2)
        else:
            final_dist = vocab_dist

        return final_dist, hidden, attn

    def forward(self, input:Tuple[Tensor,Tuple])->Tuple[Tensor,Tensor,dict]:
        raw_outputs, encoder_outputs, encoder_hidden, inputs, enc_batch_extend_vocab, extra_zeros, dec_padding_mask, dec_lens = input
        outputs = encoder_outputs
        # encoder_outputs, encoder_hidden = input

        # print('encoder outputs shape', get_shape(encoder_outputs))
        # print('encoder hidden shape', get_shape(encoder_hidden))

        encoder_outputs = encoder_outputs[-1]
        # encoder_hidden = encoder_hidden[-1]

        # encoder_hidden = torch.stack(tuple(torch.cat(tuple(e)) for e in encoder_hidden))
        # encoder_hidden = [e for e in encoder_hidden.transpose(0, 1).contiguous()]

        # print('encoder outputs shape', get_shape(encoder_outputs))
        # print('encoder hidden shape', get_shape(encoder_hidden))

        # print(encoder_hidden.shape)
        # encoder_hidden = encoder_hidden[:,:,-1,:]
        # inputs = torch.tensor([self.eos_id]*encoder_hidden[0].shape[0], device=defaults.device)
        # inputs = None
        function = F.log_softmax
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, self.teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            initial_input = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if defaults.device == 'cuda' and torch.cuda.is_available():
                initial_input = initial_input.cuda()
            decoder_input = torch.cat((initial_input, decoder_input), dim=1)

            for di in range(max_length):
                # print(get_shape(decoder_input))
                # print(batch_size)
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input[:,di].clone().unsqueeze(1),
                                                                              decoder_hidden, encoder_outputs,
                                                                              enc_batch_extend_vocab, extra_zeros,
                                                                              function=function)
                step_output = decoder_output.squeeze(1)
                decode(di, step_output, step_attn)
        else:
            # decoder_input = inputs[:, 0].unsqueeze(1)
            initial_input = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if defaults.device == 'cuda' and torch.cuda.is_available():
                initial_input = initial_input.cuda()
            decoder_input = initial_input
            # decoder_input = torch.cat((initial_input, decoder_input), dim=1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                              encoder_outputs, enc_batch_extend_vocab,
                                                                              extra_zeros, function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        decoder_outputs = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
        # print('decoder_outputs', get_shape(decoder_outputs))

        # print(get_shape(decoder_outputs))
        return [decoder_outputs, dec_padding_mask, dec_lens], raw_outputs, outputs

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1)# - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

# TODO: implement this function.
def get_seq2seq(vocab_sz_inp:int, vocab_sz_out:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, max_len:int,
                tie_weights:bool=True, hidden_size=1150, bidir:bool=False, qrnn:bool=False,
                hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5,
                teacher_forcing_ratio:float=0)->nn.Module:
    "Create a seq2seq model."
    rnn_enc = RNNCore(vocab_sz_inp, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, qrnn=qrnn, bidir=bidir,
                      hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    # enc = rnn_enc.encoder if tie_weights else None # TODO: tie encoder embedding weights to decoder last layer.
    # rnn_enc = RNNCore(bptt, max_seq, vocab_sz, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
    #                   qrnn=qrnn, hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    rnn_dec = DecoderRNN(vocab_sz_out, emb_sz, max_len, hidden_size, 0, 1, n_layers, 'lstm', bidir, input_p, 0.1, True,
                         teacher_forcing_ratio=teacher_forcing_ratio) # bptt, max_seq, hidden_size, vocab_sz, dropout_p=0.1, max_length=10)
    model = SequentialRNN(rnn_enc, rnn_dec)
    model.reset()
    return model