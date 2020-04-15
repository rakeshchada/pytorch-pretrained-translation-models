from transformer import TransformerModel
import numpy as np
from copy import copy
import torch
import random
import argparse

def set_seed():
    SEED = 1111
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


transformer_param_mappings = lambda i: {
    # encoder level mappings
    f'transformer.encoder.layers.{i}.self_attn.in_proj_weight': [f'encoder_l{i+1}_self_Wq', f'encoder_l{i+1}_self_Wk', f'encoder_l{i+1}_self_Wv'],
    f'transformer.encoder.layers.{i}.self_attn.in_proj_bias': [f'encoder_l{i+1}_self_bq', f'encoder_l{i+1}_self_bk', f'encoder_l{i+1}_self_bv'],
    f'transformer.encoder.layers.{i}.self_attn.out_proj.weight': f'encoder_l{i+1}_self_Wo',
    f'transformer.encoder.layers.{i}.self_attn.out_proj.bias': f'encoder_l{i+1}_self_bo',
    f'transformer.encoder.layers.{i}.norm1.weight': f'encoder_l{i+1}_self_Wo_ln_scale',
    f'transformer.encoder.layers.{i}.norm1.bias': f'encoder_l{i+1}_self_Wo_ln_bias',
    f'transformer.encoder.layers.{i}.linear1.weight': f'encoder_l{i+1}_ffn_W1',
    f'transformer.encoder.layers.{i}.linear1.bias': f'encoder_l{i+1}_ffn_b1',
    f'transformer.encoder.layers.{i}.linear2.weight': f'encoder_l{i+1}_ffn_W2',
    f'transformer.encoder.layers.{i}.linear2.bias': f'encoder_l{i+1}_ffn_b2',
    f'transformer.encoder.layers.{i}.norm2.weight': f'encoder_l{i+1}_ffn_ffn_ln_scale',
    f'transformer.encoder.layers.{i}.norm2.bias': f'encoder_l{i+1}_ffn_ffn_ln_bias',
    # decoder level mappings
    f'transformer.decoder.layers.{i}.self_attn.in_proj_weight': [f'decoder_l{i+1}_self_Wq', f'decoder_l{i+1}_self_Wk', f'decoder_l{i+1}_self_Wv'],
    f'transformer.decoder.layers.{i}.self_attn.in_proj_bias': [f'decoder_l{i+1}_self_bq', f'decoder_l{i+1}_self_bk', f'decoder_l{i+1}_self_bv'],
    f'transformer.decoder.layers.{i}.self_attn.out_proj.weight': f'decoder_l{i+1}_self_Wo',
    f'transformer.decoder.layers.{i}.self_attn.out_proj.bias': f'decoder_l{i+1}_self_bo',
    f'transformer.decoder.layers.{i}.multihead_attn.in_proj_weight': [f'decoder_l{i+1}_context_Wq', f'decoder_l{i+1}_context_Wk', f'decoder_l{i+1}_context_Wv'],
    f'transformer.decoder.layers.{i}.multihead_attn.in_proj_bias': [f'decoder_l{i+1}_context_bq', f'decoder_l{i+1}_context_bk', f'decoder_l{i+1}_context_bv'],
    f'transformer.decoder.layers.{i}.multihead_attn.out_proj.weight': f'decoder_l{i+1}_context_Wo',
    f'transformer.decoder.layers.{i}.multihead_attn.out_proj.bias': f'decoder_l{i+1}_context_bo',
    f'transformer.decoder.layers.{i}.norm1.weight': f'decoder_l{i+1}_self_Wo_ln_scale',
    f'transformer.decoder.layers.{i}.norm1.bias': f'decoder_l{i+1}_self_Wo_ln_bias',
    f'transformer.decoder.layers.{i}.norm2.weight': f'decoder_l{i+1}_context_Wo_ln_scale',
    f'transformer.decoder.layers.{i}.norm2.bias': f'decoder_l{i+1}_context_Wo_ln_bias',
    f'transformer.decoder.layers.{i}.norm3.weight': f'decoder_l{i+1}_ffn_ffn_ln_scale',
    f'transformer.decoder.layers.{i}.norm3.bias': f'decoder_l{i+1}_ffn_ffn_ln_bias',
    f'transformer.decoder.layers.{i}.linear1.weight': f'decoder_l{i+1}_ffn_W1',
    f'transformer.decoder.layers.{i}.linear1.bias': f'decoder_l{i+1}_ffn_b1',
    f'transformer.decoder.layers.{i}.linear2.weight': f'decoder_l{i+1}_ffn_W2',
    f'transformer.decoder.layers.{i}.linear2.bias': f'decoder_l{i+1}_ffn_b2'
}

def generate_all_param_mappings(nlayers, pretrained_weights):
    param_mappings = {}

    # Transformer level mappings
    param_mappings['encoder.weight'] = pretrained_weights['Wemb']
    param_mappings['decoder.weight'] = pretrained_weights['Wemb']
    param_mappings['fc_out.weight'] = pretrained_weights['Wemb']
    param_mappings['fc_out.bias'] = pretrained_weights['decoder_ff_logit_out_b'][0, :]

    keys_visited = list(pretrained_weights.keys())

    keys_visited.remove('Wemb')
    keys_visited.remove('decoder_ff_logit_out_b')
    for i in range(nlayers):
        for new_model_param_name, pretrained_param_name in transformer_param_mappings(i).items():
            if isinstance(pretrained_param_name, list):
                if 'bias' in new_model_param_name:
                    param_value_to_copy = np.concatenate([pretrained_weights[ppn][0, :] for ppn in pretrained_param_name])
                else:
                    param_value_to_copy = np.concatenate([np.transpose(pretrained_weights[ppn]) for ppn in pretrained_param_name])
                for ppn in pretrained_param_name:
                    keys_visited.remove(ppn)
            elif 'bias' in new_model_param_name or 'norm' in new_model_param_name:
                param_value_to_copy = pretrained_weights[pretrained_param_name][0, :]
                keys_visited.remove(pretrained_param_name)     
            else:
                param_value_to_copy = np.transpose(pretrained_weights[pretrained_param_name])
                keys_visited.remove(pretrained_param_name)
            
            param_mappings[new_model_param_name] = param_value_to_copy

    print(f"***************WARN****************Unused pretrained parameters: {keys_visited}")
                
    return param_mappings

def copy_weights(model, param_mappings):
    with torch.no_grad():
        for pname, param in model.named_parameters():
            if pname not in param_mappings:
                print(f"***************WARN****************{pname} not found in pre-defined parameter mappings.")
            pretrained_param_weights = param_mappings[pname]
            param.copy_(torch.from_numpy(pretrained_param_weights).float())

# TODO: This causes several warnings about duplicate keys. So it might be messing up the indexes.
# The dictionary it returns is of lower size than vocab size due to this reason.
def parse_vocab_file(filepath):
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    yaml.allow_duplicate_keys = True
    spiece_ids = yaml.load(open(filepath))
    return spiece_ids


def load_transformer_weights(model, pretrained_weights_path):
    model.encoder.weight = model.decoder.weight = model.fc_out.weight

    pretrained_weights = np.load(pretrained_weights_path)

    all_param_mappings = generate_all_param_mappings(nlayers, pretrained_weights)

    copy_weights(model, all_param_mappings)

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="Path to the directory of the model",
    )
    parser.add_argument(
        "--weights_file_name",
        default='opus.spm32k-spm32k.transformer-align.model1.npz.best-perplexity.npz',
        type=str,
        required=True,
        help="Name of the file containing model weights",
    )
    parser.add_argument(
        "--vocab_file_name",
        default='opus.spm32k-spm32k.vocab.yml',
        type=str,
        required=True,
        help="Name of the vocab file",
    )
    parser.add_argument(
        "--source_tokenizer_model",
        default='source.spm',
        type=str,
        required=True,
        help="Name of source tokenizer model file",
    )
    parser.add_argument(
        "--target_tokenizer_model",
        default='target.spm',
        type=str,
        required=True,
        help="Name of the target tokenizer model file",
    )
    parser.add_argument(
        "--vocab_size",
        default=62521,
        type=int,
        help="Size of the vocabulary",
    )
    parser.add_argument(
        "--text_to_translate",
        default='hey there',
        type=str,
        help="Text to input to the translation model",
    )

    args = parser.parse_args()


    set_seed()

    d_model = 512
    d_ff = 2048
    nlayers = 6
    nheads = 8
    device = "cpu"
    src_vocab_size = trg_vocab_size = args.vocab_size
    weights_path = f"{args.model_dir}/{args.weights_file_name}"
    vocab_path = f"{args.model_dir}/{args.vocab_file_name}"
    source_tokenizer_model_path = f"{args.model_dir}/{args.source_tokenizer_model}"
    target_tokenizer_model_path = f"{args.model_dir}/{args.target_tokenizer_model}"


    # Load Vocab
    spiece_ids = parse_vocab_file(vocab_path)

    spiece_id_to_tokens = {v: k for k, v in spiece_ids.items()}

    eos_token = '</s>'
    eos_id = spiece_ids[eos_token]

    import sentencepiece as spm

    src_tokenizer = spm.SentencePieceProcessor()
    src_tokenizer.Load(source_tokenizer_model_path)

    trg_tokenizer = spm.SentencePieceProcessor()
    trg_tokenizer.Load(target_tokenizer_model_path)

    model = TransformerModel(intoken=src_vocab_size, outtoken=trg_vocab_size, hidden=d_model, 
        d_ff=d_ff, nlayers=nlayers, dropout=0.0).to(device)

    model = load_transformer_weights(model, weights_path)
    print(f"Done loading pretrained model weights!")

    model.eval()

    src_pieces = src_tokenizer.encode_as_pieces(args.text_to_translate)
    src_input_ids = [spiece_ids[piece] for piece in src_pieces] + [eos_id]

    src_tensor = torch.LongTensor(src_input_ids).unsqueeze(1).to(device)

    trg_indexes = [eos_id]

    max_len = 25

    with torch.no_grad():

        for i in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)

            output = model(src_tensor, trg_tensor)
            
            pred_token = output.argmax(-1)[-1,:].item()
            
            trg_indexes.append(pred_token)

            if pred_token == eos_token:
                break

    predicted_translation = " ".join([spiece_id_to_tokens[idx] for idx in trg_indexes])
    print(f"Predicted Translation: {predicted_translation}")
