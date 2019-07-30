import os
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Dense, LeakyReLU, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

def read_cupt_file(cupt_file):
    """Reads in cupt file.
    
    Returns a lists of sentences. The sentences are represented by lists, the
    words by tuples in those lists: 
    [('1', 'Mecklenburg', 'Mecklenburg', 'PROPN', 'NE',
    '_', '7', 'nsubj', '_', '_', '*'),...]
    """
    f = open(cupt_file, 'r', encoding='utf-8')
    sents = f.readlines()
    f.close()
    sents_no_comments = [line for line in sents if not line.startswith('#')]
    sents_split = []
    sent = []
    for line in sents_no_comments:
        if line not in ['\n']:
            if line.split('\t')[0].isdigit():
                sent.append(tuple(line.rstrip('\n').split('\t')))
        else:
            sents_split.append(sent)
            sent = []
    return sents_split

def read_raw_cupt_file(cupt_file):
    f = open(cupt_file, 'r', encoding='utf-8')
    sents = f.readlines()
    f.close()
    sents_split = []
    sent = []
    for line in sents:
        if line not in ['\n']:
            if line.startswith('#'):
                sent.append(line)
            else:
                sent.append(list(line.split('\t')))
        else:
            sents_split.append(sent)
            sent = []
    return sents_split

def convert_to_dfs(list_sents):
    """Converts the sentences represented by lists to pandas DataFrames."""
    df_sents = [pd.DataFrame(sent, columns=['ID', 'FORM', 'LEMMA', 'UPOS',
                                            'XPOS', 'FEATS', 'HEAD', 'DEPREL',
                                            'DEPS', 'MISC', 'PARSEME:MWE'])
               for sent in list_sents]
    for df_sent in df_sents:
        df_sent['SINGLE_LABEL'] = '0'
    return df_sents

def get_tags(df_sents):
    """Returns a list of all tags."""
    tags = [tag for sent in df_sents for tag in list(sent['PARSEME:MWE'])]
    tags = list(set(tags))
    return tags

def set_single_label(label_type, df_sent):
    """Fills the SINGLE_LABEL column of a sentence represented as a DataFrame.
    
    Takes as input a label type, e.g. LVC, and a sentence. It searches for 
    the rows that are labeled with the particular label type and fills the
    SINGLE_LABEL cell in that row with a 1.
    """
    mwe_ids = []
    labeled_rows = df_sent[df_sent['PARSEME:MWE'].str.contains(label_type)]
    for labels in labeled_rows['PARSEME:MWE']:
        split_labels = labels.split(';')
        for label in split_labels:
            if label_type in label:
                mwe_id = label.split(':')[0]
                mwe_ids.append(mwe_id)
    for i, row in df_sent.iterrows():
        row_labels = []
        parseme_labels = row[10]
        parseme_labels = parseme_labels.split(';')
        for label in parseme_labels:
            label_id = label.split(':')[0]
            if label_id in mwe_ids:
                row_labels.append(label_id)
        if row_labels:
            df_sent.loc[i, ['SINGLE_LABEL']] = ';'.join(row_labels)

def set_edge_label(train_sent_graph):
    """Sets the edge label."""
    for edge in train_sent_graph.edges.items():
        head = edge[0][0]
        dep = edge[0][1]
        head_label = train_sent_graph.node[head]['attr']['node_label'].split(';')
        dep_label = train_sent_graph.node[dep]['attr']['node_label'].split(';')
        if not set(head_label).isdisjoint(set(dep_label)) and \
           (head_label[0] != '0' and dep_label[0] != '0'):
            edge[1]['edge_label'] = 1

def build_sent_graph(df_sent):
    """Builds the dependency graph of a sentence."""
    sent_graph = nx.OrderedDiGraph()
    for index, row in df_sent.iterrows():
        try:
            dep_id, head_id, dep_form = row[0], row[6], row[1]
            pos, deprel = row[3], row[7]
            node_label = row[11]
            sent_graph.add_node(dep_id,
                                attr={'form': dep_form, 'pos': pos,
                                      'deprel': deprel,
                                      'node_label': node_label})
            sent_graph.add_edge(dep_id, head_id, edge_label='0')
        except IndexError as e:
            print(e)
    sent_graph.node['0']['attr'] = {'form': 'DUMMY-ROOT-FORM',
                                    'pos': 'DUMMY-ROOT-POS', 'deprel': 'root',
                                    'node_label': '0'}
    return sent_graph

def get_feature_dicts(*data_sets):
    """Returns the form, pos and deprel indeces."""
    df_sents = []
    for data_set in data_sets:
        df_sents.extend(data_set)
    forms = list({form for sent in df_sents for form in list(sent['FORM'])})
    pos_tags = list({pos for sent in df_sents for pos in list(sent['UPOS'])})
    deprels = list({deprel for sent in df_sents
                    for deprel in list(sent['DEPREL'])})
    forms_to_id = {form: idx + 1 for idx, form in enumerate(forms)}
    pos_to_id = {pos: idx + 1 for idx, pos in enumerate(pos_tags)}
    deprel_to_id = {deprel: idx for idx, deprel in enumerate(deprels)}
    forms_to_id['DUMMY-ROOT-FORM'] = 0
    pos_to_id['DUMMY-ROOT-POS'] = 0
    return forms_to_id, pos_to_id, deprel_to_id

def build_X_and_y(form_to_id, pos_to_id, deprel_to_id, sent_graphs):
    """Returns X_form, X_pos, X_deprels and y.
    
    Builds the label vector y and the data matrices for word forms, POS tags, 
    dependency and relations. 
    """
    X_forms, X_pos, X_deprels = [], [], []
    y = []
    for sent_graph in sent_graphs:
        for edge in sent_graph.edges.items():
            head = edge[0][0]
            dep = edge[0][1]
            label = edge[1]['edge_label']
            # Head features
            head_form = sent_graph.node[head]['attr']['form']
            head_pos = sent_graph.node[head]['attr']['pos']
            head_deprel = sent_graph.node[dep]['attr']['deprel']
            # Dep features
            dep_form = sent_graph.node[dep]['attr']['form']
            dep_pos = sent_graph.node[dep]['attr']['pos']
            dep_deprel = sent_graph.node[dep]['attr']['deprel']
            # Adding training examples to the different X matrices
            X_forms.append([form_to_id[head_form], form_to_id[dep_form]])
            X_pos.append([pos_to_id[head_pos], pos_to_id[dep_pos]])
            X_deprels.append([deprel_to_id[head_deprel],
                              deprel_to_id[dep_deprel]])
            y.append(label)
    assert len(X_forms) == len(y)
    return np.array(X_forms), np.array(X_pos), np.array(X_deprels), np.array(y)

def build_embedding_matrix(emb_dir, form_to_id):
    """Returns the fasttext embedding matrix for word forms."""
    emb_index = {}
    for dir_path, dir_names, file_names in os.walk(emb_dir):
        for file_name in file_names:
            emb_path = os.path.join(dir_path, file_name)
            f = open(emb_path, 'r', encoding='utf-8')
            try:
                for line in f:
                    if line not in ['\n']:
                        values = line.split()
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        emb_index[word] = coefs
            except UnicodeDecodeError as e:
                print('test')
            f.close()
    print('Found {} word vectors.'.format(len(emb_index)))
    
    emb_matrix = np.zeros((len(form_to_id) + 1, 300))
    for form, id in form_to_id.items():
        emb_vector = emb_index.get(form)
        if emb_vector is not None:
            emb_matrix[id] = emb_vector
    print('Embeddings matrix has shape {}.'.format(emb_matrix.shape))
    
    return emb_matrix

def add_pred_labels_to_graphs(graphs_no_labels, predicted_labels):
    """Adds the prediction labels to the graphs."""
    edges_list = []
    for graph in graphs_no_labels:
        for edge in graph.edges.items():
            edges_list.append(edge)
    for edge, label in zip(edges_list, predicted_labels):
        edge[1]['edge_label'] = label

def decode_labels(labeled_graph):
    """Decoding the labels."""
    mwe_labels = []
    for graph in labeled_graph:
        for edge in graph.edges.items():
            if edge[1]['edge_label'] == 0:
                graph.remove_edge(*edge[0])
        connected_nodes = nx.connected_components(graph.to_undirected())
        con_nodes_list = []
        mwe_id = 1
        for cn in connected_nodes:
            if len(cn) > 1:
                con_nodes_list.append((mwe_id, cn))
                mwe_id += 1
        mwe_labels.append(con_nodes_list)
        mwe_id = 1
    return mwe_labels

def tag_cupt_file(raw_file, mwe_labels, mwe_type, output_name):
    """Transferring the predictions for the first VMWE type to a cupt file."""
    f = open(output_name, 'w', encoding='utf-8')
    for sent, ml in zip(raw_file, mwe_labels):
        id_tracker = []
        for line in sent:
            if not isinstance(line, list):
                f.write(line)
            else:
                label_id = '*'
                token_id = line[0]
                for label in ml:
                    if token_id in label[1]:
                        label_id = str(label[0])
                        id_tracker.append(label_id)
                        break
                columns = '\t'.join(line[:-1])
                if id_tracker.count(label_id) == 1:
                    f.write(columns + '\t' + label_id + ':' + mwe_type + '\n')
                else:
                    f.write(columns + '\t' + label_id + '\n')
        f.write('\n')
    f.close()

def main():
    parser = argparse.ArgumentParser(description='Local Model')
    parser.add_argument('--lang_dir', type=str,
                        help='Path to language directory of a parseme dataset')
    parser.add_argument('--train', type=str,
                        help='Path to train file.')
    parser.add_argument('--dev', type=str,
                        help='Path to dev file.')
    parser.add_argument('--test', type=str,
                        help='Path to test file.')
    parser.add_argument('--mwe_type', type=str,
                        help='MWE type the classifier is trained on.')
    parser.add_argument('--embs', type=str,
                        help='Path to the word embeddings.')
    args = parser.parse_args()
    
    with open('configs/params.json') as f:
        config = json.load(f)

    # Read data
    list_train_sents = read_cupt_file(args.train)
    list_dev_sents = read_cupt_file(args.dev)
    list_test_sents = read_cupt_file(args.test)

    # Convert sentences in data to DataFrame objects
    df_train_sents = convert_to_dfs(list_train_sents)
    df_dev_sents = convert_to_dfs(list_dev_sents)
    df_test_sents = convert_to_dfs(list_test_sents)

    print("The train set contains {} sentences.".format(len(df_train_sents)))
    print("The dev set contains {} sentences.".format(len(df_dev_sents)))
    print("The test set contains {} sentences.".format(len(df_test_sents)))

    tags = get_tags(df_train_sents)

    form_to_id, pos_to_id, deprel_to_id = get_feature_dicts(df_train_sents,
                                                            df_dev_sents,
                                                            df_test_sents)

    form_num = len(form_to_id)
    pos_num = len(pos_to_id)
    deprel_num = len(deprel_to_id)

    print("The word index contains {} different forms.".format(form_num))
    print("The pos index contains {} different POS.".format(pos_num))
    print("The deprel index contains {}"
          " different dependency relations.".format(deprel_num))

    for df_train_sent in df_train_sents:
        set_single_label(args.mwe_type, df_train_sent)
    for df_dev_sent in df_dev_sents:
        set_single_label(args.mwe_type, df_dev_sent)
    for df_test_sent in df_test_sents:
        set_single_label(args.mwe_type, df_test_sent)

    print("Building graphs...")

    train_sent_graphs = [build_sent_graph(df_train_sent)
                         for df_train_sent in df_train_sents]
    dev_sent_graphs = [build_sent_graph(df_dev_sent)
                       for df_dev_sent in df_dev_sents]
    test_sent_graphs = [build_sent_graph(df_test_sent)
                        for df_test_sent in df_test_sents]

    dev_sent_graphs_no_labels = [build_sent_graph(df_dev_sent)
                                 for df_dev_sent in df_dev_sents]
    test_sent_graphs_no_labels = [build_sent_graph(df_test_sent)
                                  for df_test_sent in df_test_sents]

    print("Set edge labels...")

    for train_sent_graph in train_sent_graphs:
        set_edge_label(train_sent_graph)
    for dev_sent_graph in dev_sent_graphs:
        set_edge_label(dev_sent_graph)
    for test_sent_graph in test_sent_graphs:
        set_edge_label(test_sent_graph)

    print("Building X and y...")

    X_forms_train, X_pos_train, X_deprels_train, y_train = build_X_and_y(
        form_to_id, pos_to_id, deprel_to_id, train_sent_graphs)
    X_forms_dev, X_pos_dev, X_deprels_dev, y_dev = build_X_and_y(
        form_to_id, pos_to_id, deprel_to_id, dev_sent_graphs)
    X_forms_test, X_pos_test, X_deprels_test, y_test = build_X_and_y(
        form_to_id, pos_to_id, deprel_to_id, test_sent_graphs)

    print(X_forms_test.shape)
    print(y_test.shape)

    y_train = to_categorical(y_train)
    y_dev = to_categorical(y_dev)
    y_test = to_categorical(y_test)

    print(y_test.shape)

    print("Buildung the matrix with pretrained embeddings...")

    emb_matrix = build_embedding_matrix(args.embs, form_to_id)

    print("The embedding matrix has shape {}".format(emb_matrix.shape))
    
    form_input = Input(shape=(2,))
    form_emb = Embedding(input_dim=form_num + 1,
                         output_dim=config['embedding_size'],
                         weights=[emb_matrix], trainable=False)(form_input)

    pos_input = Input(shape=(2,))
    pos_emb = Embedding(input_dim=pos_num,output_dim=config['pos_emb_size'],
                        input_length=2)(pos_input)

    deprel_input = Input(shape=(2,))
    deprel_emb = Embedding(input_dim=deprel_num,
                           output_dim=config['deprel_emb_size'],
                           input_length=2)(deprel_input)

    conc_x = concatenate([form_emb, pos_emb, deprel_emb])

    flat_conc = Flatten()(conc_x)

    hidden = Dense(200)(flat_conc)
    leaky_layer = LeakyReLU(alpha=0.05)(hidden)
    output = Dense(2, activation='softmax')(leaky_layer)

    local_model = Model(inputs=[form_input, pos_input, deprel_input],
                        outputs=output)

    local_model.summary()

    local_model.compile(optimizer=Adam(lr=config['adam']['learning_rate'],
                                       beta_1=config['adam']['beta_1'],
                                       beta_2=config['adam']['beta_2'],
                                       epsilon=config['adam']['epsilon'],
                                       decay=config['adam']['decay'],
                                       amsgrad=config['adam']['amsgrad']),
                        loss=config['loss'], metrics=[config['metrics']])

    history = local_model.fit([X_forms_train, X_pos_train, X_deprels_train],
                              y_train, batch_size=config['batch_size'],
                              epochs=config['epochs'],
                              validation_split=0.1, verbose=1)
    
    dev_predictions = local_model.predict([X_forms_dev, X_pos_dev,
                                           X_deprels_dev])
    dev_predicted_labels = np.argmax(dev_predictions, axis=1)
    test_predictions = local_model.predict([X_forms_test, X_pos_test,
                                            X_deprels_test])
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    
    y_dev = np.argmax(y_dev, axis=1)
    y_test = np.argmax(y_test, axis=1)

    add_pred_labels_to_graphs(dev_sent_graphs_no_labels, dev_predicted_labels)
    add_pred_labels_to_graphs(test_sent_graphs_no_labels, test_predicted_labels)

    dev_sent_graphs_labeled = dev_sent_graphs_no_labels.copy()
    test_sent_graphs_labeled = test_sent_graphs_no_labels.copy()

    print("Decoding the labels predicted by the local model...")

    dev_mwe_labels = decode_labels(dev_sent_graphs_labeled)
    test_mwe_labels = decode_labels(test_sent_graphs_labeled)

    raw_dev_file = read_raw_cupt_file(args.dev)
    raw_test_file = read_raw_cupt_file(args.test)

    print("Tagging the cupt files...")

    dev_pred_file = args.mwe_type + '_dev_preds.cupt'
    test_pred_file = args.mwe_type + '_test_preds.cupt'

    tag_cupt_file(raw_dev_file, dev_mwe_labels, args.mwe_type,
                        dev_pred_file)
    tag_cupt_file(raw_test_file, test_mwe_labels, args.mwe_type,
                        test_pred_file)
    
if __name__ == '__main__':
    main()

