import json
from torch.utils.data import Dataset


class TextSummaryDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'abstractive': self.data['abstractive'],
            'article_original': self.data['article_original'],
            'extractive': self.data['extractive'],
            'id': self.data['id'],
            'media': self.data['media']
        }
        return sample

# import time
# s_time = time.time()
# dataset = TextSummaryDataset('data/train.jsonl')
# print(time.time() - s_time)