import argparse

from custom_layers import AverageWords
from preprocess import PreProcessor

from keras.layers import Embedding, Dense, Input, BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad

embedding_dim = 300
num_hidden_layers = 3
num_hidden_units = 300
num_epochs = 50
batch_size = 100
dropout_rate = 0.5
activation = 'relu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', help='location of dataset', default='data/out_split.pk')
    parser.add_argument('-data', help='location of dataset', default='data/out_sent_split.pk')
    parser.add_argument('-We', help='location of word embeddings', default='data/glove.6B.300d.txt')
    parser.add_argument('-model', help='model to run: nbow or dan', default='nbow')

    args = vars(parser.parse_args())

    pp = PreProcessor(args['data'],args['We'])
    pp.tokenize()
    data, labels, data_val, labels_val = pp.make_data()
    embedding_matrix = pp.get_word_embedding_matrix()

    model = Sequential()
    model.add(Embedding(len(pp.word_index)+1,embedding_dim,weights=[embedding_matrix],\
        input_length=pp.MAX_SEQUENCE_LENGTH))
    model.add(AverageWords())

    if args['model'] == 'dan':
        for i in range(num_hidden_layers):
            model.add(Dense(num_hidden_layers))
            model.add(BatchNormalization())
            model.add(Activation(activation))
            model.add(Dropout(dropout_rate))

    model.add(Dense(labels.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Activation('softmax'))

    adagrad = Adagrad()
    model.compile(loss='categorical_crossentropy',optimizer=adagrad,metrics=['categorical_accuracy'])

    model.summary()

    model.fit(data,labels,batch_size=batch_size,epochs=num_epochs,validation_data=(data_val,labels_val))
