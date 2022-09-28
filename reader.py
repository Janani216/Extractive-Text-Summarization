import glob
import re
import pandas as pd


class DataReader:
    def __init__(self, path):
        self.path = path
        self.documents = {}
        self.summaries = {}

    def load_dataset(self):
        document_files = sorted(glob.glob(self.path + "/document*.txt"))
        summary_files = sorted(glob.glob(self.path + "/summary*.txt"))
        for document_file in document_files:
            doc_num = re.findall(r'\d+', document_file)[0]
            with open(document_file, 'r') as file:
                data = file.read().replace('\n', '')
            self.documents[doc_num] = data
        for summary_file in summary_files:
            doc_num = re.findall(r'\d+', summary_file)[0]
            with open(summary_file, 'r') as file:
                data = file.read().replace('\n', '')
            self.summaries[doc_num] = data
        return self.documents, self.summaries

    def load_dataset_from_csv(self):
        df = pd.read_csv(self.path,nrows = 1000,header=None)
        return df[df.columns[1]].to_list(), df[df.columns[2]].to_list()
