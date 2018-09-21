import fastgcn as fg
import numpy as np
import tensorflow as tf

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
adjacent_matrix = np.array(data_processor.get_adjacent_matrix(),dtype="float32")

placeholders = {}
placeholders.update({"support":adjacent_matrix})

model = fg.Sequential(name="fg",placeholders=placeholders,sample_num=100)
model.add(fg.Dense(input_shape=inputs.shape,placeholders=placeholders,output_dim=inputs.shape[1]))
model.add(fg.GraphConvolution(input_shape=inputs.shape,support=None,placeholders=placeholders,output_dim=[7],dropout=0.))
model.compile(loss="masked_softmax_cross_entropy",metrics=["accuracy"])
model.fit(x=inputs[:,:],y=labels[:,:],epochs=50,batch_size=100)