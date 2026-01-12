def load_data_generator():
    from datasets import load_dataset
    stream_data = load_dataset(path="data/Pep2Prob/data_split/train_test_split_set_1", streaming=True)
    return stream_data
