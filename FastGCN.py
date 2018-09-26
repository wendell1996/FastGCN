import fastgcn as fg
import numpy as np
import tensorflow as tf
import scipy.io
""""
data_content_file_path = "data/cora/cora.content"
data_cites_file_path = "data/cora/cora.cites"
write_graphML_path = "data/cora/cora.graphml"
write_gexf_path = "data/cora/cora.gexf"
dataset_name = "cora"
data_processor = fg.utils.DataProcessor(data_content_file_path=data_content_file_path, data_cites_file_path=data_cites_file_path, dataset_name=dataset_name)
dataset = data_processor()
#data_processor.write_graphML(write_graphML_path=write_graphML_path)
#data_processor.write_gexf(write_gexf_path=write_gexf_path)
inputs,labels = data_processor.get_data()
adjacent_matrix_all = np.array(data_processor.get_adjacent_matrix(),dtype="float32")
adjacent_matrix = adjacent_matrix_all[:1208,:1208]
placeholders = {}
placeholders.update({"support":adjacent_matrix})
"""
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

def load_mat_data(path="data.mat"):
    data = scipy.io.loadmat(path)
    features = data["features"]
    train_mask = data["train_mask"]
    validation_mask = data["val_mask"]
    test_mask = data["test_mask"]
    labels_train = data["y_train"]
    labels_validation = data["y_val"]
    labels_test = data["y_test"]
    adjacent_matrix_all = data["adj"]
    return adjacent_matrix_all,features,train_mask,validation_mask,test_mask,labels_train,labels_validation,labels_test

def main(ranks=[50,50]):
    adjacent_matrix_all,features,train_mask,validation_mask,test_mask,y_train,y_validation,y_test = load_mat_data()
    train_index = np.where(train_mask)[1]
    validation_index = np.where(validation_mask)[1]
    test_index = np.where(test_mask)[1]
    features_train = features[train_index,:]
    features_validation = features[validation_index,:]
    features_test = features[test_index,:]
    labels_train = y_train[train_index,:]
    labels_validation = y_validation[validation_index,:]
    labels_test = y_test[test_index,:]
    adjacent_matrix_train = adjacent_matrix_all[train_index,:][:,train_index]
    adjacent_matrix_validation = adjacent_matrix_all[validation_index,:][:,validation_index]
    placeholders = {"supports":[adjacent_matrix_train]}
    model = fg.Sequential(name="fg")
    #model.add(fg.Dense(input_shape=inputs.shape,output_dim=1200))
    model.add(fg.GraphConvolution(input_shape=features_train.shape,output_dim=[16],activation=tf.nn.relu))
    model.add(fg.GraphConvolution(output_dim=[7],activation=lambda x:x))
    model.compile(losses=["softmax_cross_entropy","l2_loss"],weight_decay=5e-4,metrics=["accuracy"],learning_rate=0.001)
    model.fit(x=features_train,y=labels_train,epochs=200,batch_size=256,ranks=ranks,placeholders=placeholders)

    placeholders.update({"supports":[adjacent_matrix_all[validation_index,:],adjacent_matrix_all]})
    model.evaluate(x=features,y=labels_validation,placeholders=placeholders)

if __name__ == "__main__":
    main([50,50])