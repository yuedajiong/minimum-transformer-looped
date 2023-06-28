import torch  #completed transformer,  looped-transformer in-development but binary_vector completed, 

class Core(torch.nn.Module):
    class Attention(torch.nn.Module):  #expanded-implemented-by-myself attention for looped-transformer
        def __init__(self, dim_in, dim_q, dim_k):
            super().__init__()
            self.q = torch.nn.Linear(dim_in, dim_q)  #TODO: qkv=nn.Linear(dim_in*3, dim_q) @PyTorch
            self.k = torch.nn.Linear(dim_in, dim_k)
            self.v = torch.nn.Linear(dim_in, dim_k)

        def forward(self, query, key, value, attn_mask, key_padding_mask, need_weights):  #not-use-so-far: attn_mask, key_padding_mask, need_weights
            def scaled_dot_product_attention(query, key, value):
                temp = query.bmm(key.transpose(1, 2))
                scale = query.size(-1) ** 0.5  #scale-or-not @loop_transformer?
                weights = torch.nn.functional.softmax(temp / scale, dim=-1)  #hardmax@loop_transformer?
                return weights.bmm(value), weights if need_weights else None
            return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

    class MultiHeadAttention(torch.nn.Module):  #No Mask for Causal
        def __init__(self, dim_in, dim_q, dim_k, num_heads):
            super().__init__()
            self.heads = torch.nn.ModuleList([Attention(dim_in, dim_q, dim_k) for _ in range(num_heads)])
            self.linear = torch.nn.Linear(num_heads * dim_k, dim_in)

        def forward(self, query, key, value, attn_mask, key_padding_mask, need_weights):
            attent_outputs = []
            attent_weights = []
            for head in self.heads:
                attent_output_weight = head(query, key, value, attn_mask, key_padding_mask, need_weights)
                attent_outputs.append(attent_output_weight[0])
                if need_weights:  #TODO refer to multi_head_attention_forward in functional.py @pytorch;   only-used-singlehead-attention in looped-transformer?
                    attent_weights.append(attent_output_weight[1])
            return self.linear(torch.cat(attent_outputs, dim=-1)), torch.mean(attn_output_weights, dim=1) if need_weights else None

    def __init__(self, d_model, nhead=None, d_feedforward=None, batch_first=None, dropout=None, norm_first=None, layer_norm_eps=0.00001, device=None, dtype=None):
        super().__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if norm_first is not None else None
        self.attent = self.__class__.Attention(d_model, d_model, d_model)  #self.__class__.MultiHeadAttention(d_model, d_model, d_model, nhead)  #torch.nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout) if dropout is not None else None
        #
        self.ffw = torch.nn.Sequential(torch.nn.Linear(d_model, d_feedforward), torch.nn.ReLU(), torch.nn.Linear(d_feedforward, d_model))
        self.dropout2 = torch.nn.Dropout(dropout) if dropout is not None else None
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if norm_first is not None else None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a = x
        if self.norm1 is not None and self.norm_first:
            a = self.norm1(a)
        a = self.attent(a, a, a, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        if self.dropout1 is not None:
            a = self.dropout1(a)
        o = x + a
        if self.norm1 is not None and not self.norm_first:
            o = self.norm1(o)
        #
        f = self.ffw(o)
        if self.dropout2 is not None:
            f = self.dropout2(f)
        o = o + f
        if self.norm2 is not None:
            o = self.norm2(o)
        return o

class Embd(torch.nn.Module):
    Encode_Type = ['embedding', 'binary_vector','normal_normal','patch_normal']   #binary_vector@looped-transformer  normal_normal@igpt  patch_normal@moco/timm
    def __init__(self, vocab_size, block_size, n_embd, dropout=0.0, device=None, dtype=None, encode_type=Encode_Type[3]):
        super().__init__()
        self.encode_type = encode_type
        if self.encode_type==self.__class__.Encode_Type[0]:
            self.wte = torch.nn.Embedding(vocab_size, n_embd).to(device)
            self.wpe = torch.nn.Embedding(block_size, n_embd).to(device)
            self.drp = torch.nn.Dropout(dropout)
        elif self.encode_type==self.__class__.Encode_Type[1]:
            self.wte = torch.nn.Embedding(vocab_size, n_embd).to(device)
        elif self.encode_type==self.__class__.Encode_Type[2]:
            self.wte = torch.randn([vocab_size, n_embd], dtype=torch.float, requires_grad=False).to(device) *0.01  #torch.normal(means, std, out=None)
            self.wpe = torch.randn([block_size, n_embd], dtype=torch.float, requires_grad=False).to(device) *0.02 
        elif self.encode_type==self.__class__.Encode_Type[3]:
            class PatchEmbed(torch.nn.Module):  #for-image
                def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768):
                    super().__init__()
                    num_patches = (img_size // patch_size) * (img_size // patch_size)
                    self.img_size = img_size
                    self.patch_size = patch_size
                    self.num_patches = num_patches
                    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                def forward(self, x):
                    B, C, H, W = x.shape
                    x = self.proj(x).flatten(2).transpose(1, 2)
                    return x
            #self.wte = PatchEmbed(img_size=224, in_chans=3, embed_dim=n_embd, patch_size=16)   #img
            self.wte = torch.nn.Embedding(vocab_size, n_embd).to(device)   #txt
            self.wpe = torch.nn.Parameter(torch.randn(1, block_size, n_embd) *0.02)
        else:
            raise

    def forward(self, idx):
        if self.encode_type==self.__class__.Encode_Type[0]:
            pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)  #[0,1,2,3,4,5,]
            tok_emb = self.wte(idx) 
            pos_emb = self.wpe(pos)
            emb = tok_emb + pos_emb   #add, not cat
            o = self.drp(emb)
        elif self.encode_type==self.__class__.Encode_Type[1]:  #for location-sensitive @looped-transformer
            def dec2bin(d, n):  #n=8: math.log(8)=2.0794415416798357 dimension
                bin_msk = 2 ** torch.arange(n - 1, -1, -1).to(d.device, d.dtype)  #torch.arange(0, n, 1).unsqueeze(0)
                bin_seq = d.unsqueeze(-1).bitwise_and(bin_msk).ne(0).type(torch.int16)  #unsqueeze(1 or -1)  
                return bin_seq
            def bin2dec(b, n):
                bin_msk = 2 ** torch.arange(n - 1, -1, -1).to(b.device, b.dtype)
                dec_num = torch.sum(bin_msk * b, -1)
                return dec_num

            batch_size, sequence_length = idx.size()[0], idx.size()[1]  #too-long-length-issue?

            #forward, +-1 binarization
            pos_idx = torch.linspace(0, sequence_length, steps=sequence_length, device=idx.device, dtype=idx.dtype, requires_grad=False).repeat(batch_size,1)
            #print('pos_idx', pos_idx.shape, pos_idx)  #(batch_size, sequence_length)

            bin_seq = dec2bin(pos_idx, sequence_length)
            #print('bin_seq', bin_seq.shape, bin_seq)  #(batch_size, sequence_length,sequence_length)
            pos_enc = torch.where(bin_seq== 0, -torch.ones_like(bin_seq), +torch.ones_like(bin_seq))  #need-not: .repeat(tok_emb.shape[0], 1, 1), because has been pos_idx in pos_idx=torch.linspace().repeat(batch_size,1)
            #print('pos_enc', pos_enc.shape, pos_enc)  #(batch_size, sequence_length,sequence_length)  #+-1  #cat this after token-embeded? or convert to shorter float?
            
            if 1:  #backward, reverse; for-later-computing 
                bin_seq = torch.where(pos_enc==-1, torch.zeros_like(pos_enc), +torch.ones_like(pos_enc))
                #print('bin_seq', bin_seq.shape, bin_seq)  #(batch_size, sequence_length,sequence_length)

                pos_idx = bin2dec(bin_seq, sequence_length)
                #print('pos_idx', pos_idx.shape, pos_idx)  #(batch_size, sequence_length)

            if 0:  #principle understand; Cauchy-Schwarz-inequality  #TODO pick one column to valid: p_i_up_t * p_i = log(n)
                pos_enc_dot = torch.bmm(pos_enc, pos_enc.permute(0, -2,-1))  #transpose/permute;  matmul->batach-matmul/bmm
                print('pos_enc_dot', pos_enc_dot.shape, pos_enc_dot)  #(batch_size, sequence_length, sequence_length)

            tok_emb = self.wte(idx) 
            #print('tok_emb', tok_emb.shape)  #(batch_size, sequence_length, embed_dimension)
            #print('pos_enc', pos_enc.shape)  #(batch_size, sequence_length, sequence_length)
            o = torch.cat((tok_emb, pos_enc), dim=-1)
            #print('o', o.shape)    #(batch_size, sequence_length, embed_dimension+sequence_length)
        elif self.encode_type==self.__class__.Encode_Type[2]:
            batch_size, sequence_length = idx.size()[0], idx.size()[1]
            pos_idx = torch.linspace(0, sequence_length, steps=sequence_length, device=idx.device, dtype=idx.dtype, requires_grad=False).repeat(batch_size,1)
            #print('idx', idx.shape)  #[3, 16]
            #print('pos_idx', pos_idx.shape)  #[3, 16]
            #print('self.wte', self.wte.shape)  #[333, 128]
            #print('self.wpe', self.wpe.shape)  #[16, 128]
            tok_emb = torch.gather(self.wte, 0, idx)  #TODO  gather-nd  https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445
            pos_emb = torch.gather(self.wpe, 0, pos_idx) 
            print('tok_emb', tok_emb.shape)  #pos_emb
            print('pos_emb', pos_emb.shape)  #pos_emb
            o = tok_emb + pos_emb   #add, not cat
        elif self.encode_type==self.__class__.Encode_Type[3]:
            batch_size, sequence_length = idx.size()[0], idx.size()[1]
            pos_idx = torch.linspace(0, sequence_length, steps=sequence_length, device=idx.device, dtype=idx.dtype, requires_grad=False).repeat(batch_size,1)
            #print('idx', idx.shape)  #[3, 16]
            #print('pos_idx', pos_idx.shape)  #[3, 16]
            print('self.wpe', self.wpe.shape)  #[16, 128]
            tok_emb = self.wte(idx)  #TODO  gather-nd  https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445
            pos_emb = self.wpe  #torch.gather(self.wpe, 0, pos_idx) 
            print('tok_emb', tok_emb.shape)  #pos_emb
            print('pos_emb', pos_emb.shape)  #pos_emb
            o = tok_emb + pos_emb   #add, not cat
            print('o', o.shape)  #[3, 16, 128]
        else:
            raise
        return o

class Task(torch.nn.Module):  #TODO some simple task to show looped-transformer
    def __init__(self, n_embd, vocab_size, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, out, decode):
        out = self.norm(out)
        logits = self.head(out)
        if not decode:
            return logits
        else:
            probs = torch.torch.nn.functional.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx_next = torch.squeeze(idx_next, dim=-1)
            return logits, idx_next

class Mind(torch.nn.Module):
    def __init__(self, n_layer, vocab_size, block_size, d_model, nhead, d_feedforward, batch_first=True, device=None):
        super().__init__()
        self.embd = Embd(vocab_size=vocab_size, block_size=block_size, n_embd=d_model).to(device)
        self.loop = torch.nn.ModuleList([Core(d_model=d_model+block_size, nhead=nhead, d_feedforward=d_feedforward, batch_first=batch_first).to(device) for _ in range(n_layer)])
        self.task = Task(n_embd=d_model+block_size, vocab_size=vocab_size).to(device)

    def forward(self, I, decode):
        H = self.embd(I)
        for core in self.loop:
            H = core(H)
        T = self.task(H, decode=decode)
        return T

def main():
    batch_size = 3
    vocab_size, block_size = 333, 16
    n_layer = 13
    d_model, nhead, d_feedforward = 128, 8, 256
    mind = Mind(n_layer, vocab_size, block_size, d_model, nhead, d_feedforward)
    optimizer = torch.optim.Adam(mind.parameters(), lr=0.001)
    X = torch.randint(batch_size, vocab_size, (batch_size, block_size))
    Y = torch.randint(batch_size, vocab_size, (batch_size, block_size))
    for epoch in range(100):
        logits = mind(X, decode=False)
        loss = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)  #loss = torch.nn.MSELoss(O,Y)
        mind.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%10==0: print('epoch=%04d  loss=%.4f'%(epoch,loss.item()))

if __name__ == '__main__':
    main()


#Scratchpad
#Memory
#Commands
#hardmax  #not-softmax
class Loop:  #@Looped-Transformer
    def test():
        X = torch.tensor([
            [0,0,0,1],  #data-read
            [0,0,0,0],  #data-write
            [0,0,0,0],  #program-counter
            [1,1,1,1],  #positional-encoding
            [0,0,0,0],  #temporary-storage
            [0,0,0,0],  #scratchpad-indicate
            ])
        Q = K = torch.tensor([0, 0, 1, 1, 0, 0])
        KX = torch.matmul(K,X)
        print('KX', KX.shape)
        KXu = KX.unsqueeze(0)
        print('KXu', KXu.shape)
        KXut = KXu.transpose(0,1)
        print('KXut', KXut.shape)
        QX = torch.matmul(Q,X)
        print('QX', QX.shape)
        QXu = QX.unsqueeze(0)
        print('QXu', QXu.shape)
        Y = torch.matmul(KXut, QXu)
        print('Y', Y.shape, Y)  #all-by-matrxi-multiply, not vector-multiply
  
