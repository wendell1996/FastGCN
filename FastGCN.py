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
adjacent_matrix = adjacent_matrix[:1000,:1000]
placeholders = {}
placeholders.update({"support":adjacent_matrix})

model = fg.Sequential(name="fg",placeholders=placeholders)
model.add(fg.Dense(input_shape=inputs.shape,placeholders=placeholders,output_dim=inputs.shape[1]))
model.add(fg.GraphConvolution(placeholders=placeholders,output_dim=[7],dropout=0.))
model.compile(loss="masked_softmax_cross_entropy",metrics=["accuracy"],learning_rate=0.00005)
model.fit(x=inputs[:1000,:],y=labels[:1000,:],epochs=50,batch_size=100,rank=100)