import fastgcn as fg
import numpy as np

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
adjacent_matrix = adjacent_matrix_all[:1000,:1000]
placeholders = {}
placeholders.update({"support":adjacent_matrix})

model = fg.Sequential(name="fg",placeholders=placeholders)
model.add(fg.Dense(input_shape=inputs.shape,output_dim=1000))
model.add(fg.GraphConvolution(output_dim=[7]))
model.compile(loss="softmax_cross_entropy",metrics=["accuracy"],learning_rate=0.0001)
model.fit(x=inputs[:1000,:],y=labels[:1000,:],epochs=50,batch_size=100,rank=100)

adjacent_matrix = adjacent_matrix_all[1001:1500,1001:1500]
placeholders.update({"support":adjacent_matrix})
model.evaluate(x=inputs[1001:1500,:],y=labels[1001:1500,:],placeholders=placeholders)