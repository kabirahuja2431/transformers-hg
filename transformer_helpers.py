import torch.nn
from layers.transformer import Transformer
from layers.transformer import GatedTransformer
from layers.transformer.transformer import TransformerDecoderWithLayer
from models import TransformerEncDecModel, TransformerDecModel
from interfaces import (
    TransformerEncDecInterface,
    TransformerDecOnlyInterface,
    TransformerLMInterface,
    TransformerPrefixLMInterface,
    TransformerDecoderLMInterface,
    TransformerEncoderCLSInterface,
    TransformerMLMInterface,
)
from models.transformer_lm import TransformerLM
from models.transformer_dec import TransformerDecoderLM
from models.transformer_enc import TransformerEncoderCLS
from models.transformer_mlm import TransformerMLM


def create_lm(
    in_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    mode="enc_dec",
    use_pos_embeddig=True,
    pos_scale=1,
    dropout=0.1,
    tied_embedding=False,
    gated_model=False,
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt")
    if mode == "enc_dec":
        # breakpoint()
        return TransformerLM(
            in_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            pos_scale=pos_scale,
            transformer=Transformer if not gated_model else GatedTransformer,
            dropout=dropout,
            tied_embedding=tied_embedding,
            **args,
        )
    elif mode == "dec":
        return TransformerDecoderLM(
            in_vocab_size,
            vec_dim,
            n_heads,
            nlayers=encoder_n_layers,
            use_pos_embeddig=use_pos_embeddig,
            **args,
        )


def create_cls(
    in_vocab_size,
    out_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    use_pos_embeddig=True,
    causal_encoder=False,
    **args
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt")
    return TransformerEncoderCLS(
        in_vocab_size,
        out_vocab_size,
        state_size=vec_dim,
        nhead=n_heads,
        nlayers=encoder_n_layers,
        use_pos_embeddig=use_pos_embeddig,
        causal_encoder=causal_encoder,
        **args,
    )


def create_mlm(
    in_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    use_pos_embedding=True,
    causal_encoder=False,
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt")
    return TransformerMLM(
        n_input_tokens=in_vocab_size,
        state_size=vec_dim,
        nhead=n_heads,
        nlayers=encoder_n_layers,
        use_pos_embedding=use_pos_embedding,
        causal_encoder=causal_encoder,
        **args,
    )


def create_model(
    in_vocab_size,
    out_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    decoder_n_layers,
    is_null_encoder=False,
    dropout=0.1,
    tied_embedding=True,
    mode="enc_dec",
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt", mode=mode)
    if is_null_encoder:
        return TransformerDecModel(
            in_vocab_size,
            out_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            tied_embedding=tied_embedding,
            dropout=dropout,
            **args,
        )
    else:
        return TransformerEncDecModel(
            in_vocab_size,
            out_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            tied_embedding=tied_embedding,
            dropout=dropout,
            **args,
        )


def create_model_interface(
    model,
    label_smoothing=0.0,
    is_null_encoder=False,
    is_lm=False,
    is_cls=False,
    is_prefix_lm=False,
):
    if is_lm:
        if not is_null_encoder:
            if not is_prefix_lm:
                return TransformerLMInterface(model, label_smoothing=label_smoothing)
            else:
                return TransformerPrefixLMInterface(
                    model, label_smoothing=label_smoothing
                )
        else:
            return TransformerDecoderLMInterface(model, label_smoothing=label_smoothing)
    elif is_cls:
        return TransformerEncoderCLSInterface(model, label_smoothing=label_smoothing)
    elif is_null_encoder:
        return TransformerDecOnlyInterface(model, label_smoothing=label_smoothing)

    else:
        return TransformerEncDecInterface(model, label_smoothing=label_smoothing)


def create_mlm_interface(model, label_smoothing=0.0):
    return TransformerMLMInterface(model, label_smoothing=label_smoothing)


#### Similar interfaces for pretrained models...
