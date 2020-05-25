def greadySearch(embed_dec_inputs, hidden):
    emb_inp = embed_dec_inputs[:, 0:1]
    for i in range(embed_dec_inputs.size(1)):
        dec_input = emb_inp
        dec_output, hidden = self.decoder(dec_input, hidden)
        # dec_output [batch_size, 1, hidden_size]
        attn, _ = self.attention(dec_output, enc_outputs, attn_mask)
        # attn [bsz, 1, attn_size[

        concat_input = t.cat([dec_output, attn], dim=2)

        concat_output = self.concat(concat_input)
        logit = self.out_proj(t.tanh(concat_output)).softmax(-1)
        dec_outputs.append(logit)

        output_symbol = logit.argmax(dim=-1)
        output_symbols.append(output_symbol)
        # random的范围为[0,1) 注意这里的问题 是<= or < 一开始这里搞错啦 写的 >=
        teacher_force = random.random() < teacher_forcing_ratio

        emb_inp = embed_dec_inputs[:, i + 1:i + 2] if teacher_force else self.embedding(
            output_symbol)

    dec_outputs = t.cat(dec_outputs, dim=1)
    output_symbols = t.cat(output_symbols, dim=1)
    return dec_outputs, output_symbols