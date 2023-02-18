import torch

class Atte(torch.nn.Module):
    def __init__(self, d_model, nhead, d_feedforward, batch_first, dropout=0.1, norm_first=True, layer_norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.mha = torch.nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        #
        self.ffw = torch.nn.Sequential(torch.nn.Linear(d_model, d_feedforward), torch.nn.ReLU(), torch.nn.Linear(d_feedforward, d_model))
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a = x
        if self.norm_first:
           a = self.norm1(a)
        a = self.mha(a, a, a, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        a = self.dropout1(a)
        o = x + a
        if not self.norm_first:
            o = self.norm1(o)
        #
        f = self.ffw(o)
        f = self.dropout2(f)
        o = o + f
        o = self.norm2(o)
        return o

class Embd(torch.nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, n_embd).to(device)
        self.wpe = torch.nn.Embedding(block_size, n_embd).to(device)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, idx, simple_encode=1):
        if simple_encode:
            pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)  #[0,1,2,3,4,5,]
        else:
            def dec2bin(d, n):  #n=8: math.log(8)=2.0794415416798357 dimension
                bin_msk = 2 ** torch.arange(n - 1, -1, -1).to(d.device, d.dtype)  #torch.arange(0, n, 1).unsqueeze(0)
                bin_seq = d.unsqueeze(-1).bitwise_and(bin_msk).ne(0).type(torch.float)  #unsqueeze(1 or -1)  type(int16) .int() .float() 
                return bin_seq
            def bin2dec(b, n):
                bin_msk = 2 ** torch.arange(n - 1, -1, -1).to(b.device, b.dtype)
                dec_num = torch.sum(bin_msk * b, -1)
                return dec_num

            print('idx', idx.shape)
            s,n = idx.size()[0], idx.size()[1]  #batch,length  #John: too-long-length-issue?

            pos_idx = torch.linspace(0, n, steps=n, device=idx.device, dtype=idx.dtype, requires_grad=False).repeat(s,1)
            print('pos_idx', pos_idx.shape, pos_idx)

            bin_seq = dec2bin(pos_idx, n)
            print('bin_seq', bin_seq.shape, bin_seq)
            pos_enc = torch.where(bin_seq== 0, -torch.ones_like(bin_seq), +torch.ones_like(bin_seq))
            print('pos_enc', pos_enc.shape, pos_enc)  #cat this after token-embeded?

            #bin_seq = torch.where(pos_enc==-1, torch.zeros_like(pos_enc), +torch.ones_like(pos_enc))
            #print('bin_seq', bin_seq.shape, bin_seq)

            pos_num = bin2dec(pos_enc, n)
            print('pos_num', pos_num.shape, pos_num)

            ##pos_enc_dot = torch.matmul(pos_enc, pos_enc.t())
            ##print('pos_enc_dot', pos_enc_dot.shape, pos_enc_dot)

            raise
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx) 
        print('pos_emb', pos_emb.shape)
        print('tok_emb', tok_emb.shape)
        if simple_encode:
            emb = tok_emb + pos_emb
        else:
            pos_emb = pos_emb.repeat(tok_emb.shape[0], 1, 1)
            emb = torch.cat((tok_emb, pos_emb), dim=-1)
        o = self.drop(emb)
        return o

class Task(torch.nn.Module):
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
        self.deep = torch.nn.ModuleList([Atte(d_model=d_model, nhead=nhead, d_feedforward=d_feedforward, batch_first=batch_first).to(device) for _ in range(n_layer)])
        self.task = Task(n_embd=d_model, vocab_size=vocab_size).to(device)

    def forward(self, I, decode):
        H = self.embd(I)
        for atte in self.deep:
            H = atte(H)
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
