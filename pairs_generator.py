import pandas as pd

def pairs_generator(path):
    data = pd.read_csv(path)
    columns = ['ID', 'names', 'names_2', 'Label']
    pairs = pd.DataFrame(columns=columns)
    ind = 0
    orginal_writers = data[data.writers == data.originals]['names'].tolist()
    for index, rows in data.iterrows():
        for i in data[data.writers == rows.originals]['names']:
            if (rows.names in orginal_writers):
                pairs = pairs.append({'ID': ind, 'names': rows.names, 'names_2': i, 'Label': 0}, ignore_index=True)
            else:
                pairs = pairs.append({'ID': ind, 'names': rows.names, 'names_2': i, 'Label': 1}, ignore_index=True)
            ind = ind + 1
    train_IDs, valid_IDs = pairs_splitter(pairs)
    return pairs, train_IDs, valid_IDs
def pairs_splitter(pairs):
    train_IDs = []
    valid_IDs = []
    for i in pairs['ID']:
        if (i % 10 == 0):
            valid_IDs.append(i)
        else:
            train_IDs.append(i)

    return train_IDs, valid_IDs
