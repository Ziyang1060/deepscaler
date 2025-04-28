import pandas as pd




if __name__=='__main__':
    # Read the dataset
    dataset_path = ''
    dataset = pd.read_parquet(dataset_path)
    chat_lst = dataset['prompt'].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    print(chat_lst)
    print('Done')