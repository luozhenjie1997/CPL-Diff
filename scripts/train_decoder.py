import torch
import random
import configparser
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel, AutoModelForMaskedLM
from CPLDiff.utils.CPLDiffDataset import XDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style
from torch import device
from torch.cuda import is_available

DEVICE = device("cuda:0" if is_available() else "cpu")
print(DEVICE)

"""Read the configuration file"""
conf = configparser.ConfigParser()
conf.read('../config.ini')
conf_dict = dict(conf.items('CPLDiff_conf'))

def train_decoder(data_type='peptide'):
    tokenizer = EsmTokenizer.from_pretrained(conf_dict['denoiser_esm_model_name'])
    esm2_model = EsmModel.from_pretrained(conf_dict['denoiser_esm_model_name'], add_pooling_layer=True).to(DEVICE)
    esm2_model.eval()

    decoder = AutoModelForMaskedLM.from_pretrained(conf_dict['denoiser_esm_model_name'], torch_dtype=torch.float, trust_remote_code=True).lm_head.to(DEVICE)

    train_dataset = XDataset(pd.read_csv('../data/train/%s.csv' % data_type))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, pin_memory=True, shuffle=True,
                                                    persistent_workers=True, num_workers=8)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-3)

    writer_epoch = 0
    x_epoch = 0
    writer = SummaryWriter(log_dir='./logs/decoder')

    for i in range(1):
        pbar = tqdm(total=len(train_data_loader), miniters=0)
        decoder.train()
        for datas in train_data_loader:
            seq_encode = tokenizer(datas['sequences'], max_length=int(conf_dict['max_length']) + 2, padding='max_length', return_tensors="pt")
            seq_ids_list, train_attention_mask = seq_encode['input_ids'].to(DEVICE), seq_encode['attention_mask'].to(DEVICE)
            with torch.no_grad():
                x0 = esm2_model(seq_ids_list, train_attention_mask).last_hidden_state

            pred_logits = decoder(x0)

            optimizer.zero_grad()
            ce_losses = F.cross_entropy(pred_logits.view(-1, pred_logits.shape[-1]), seq_ids_list.view(-1), reduce=False)
            # ignore PAD token
            ce_losses = ce_losses * train_attention_mask.reshape(-1)
            loss = torch.sum(ce_losses) / torch.sum(train_attention_mask)
            loss.backward()
            optimizer.step()

            x_epoch += 1

            pbar.update(1)

            pbar.set_description_str("epoch:%s   " % (i + 1) + Fore.RED + "loss:%.4f" % loss.item() + Style.RESET_ALL)
            writer.add_scalar("train/loss", loss.item(), writer_epoch)
            writer_epoch += 1
        pbar.close()

        torch.save(decoder.state_dict(), "../save_model/%s_decoder_model_%s.pkl" % (data_type, i + 1))


if __name__ == '__main__':
    torch.manual_seed(int(conf_dict['seed']))
    torch.cuda.manual_seed_all(int(conf_dict['seed']))
    np.random.seed(int(conf_dict['seed']))
    random.seed(int(conf_dict['seed']))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_decoder(data_type='antimicrobial')
