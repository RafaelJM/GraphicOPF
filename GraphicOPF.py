import struct
import networkx as nx
import matplotlib.pyplot as plt
import math

def ReadSubgraph(file):
    Subgraph = {}
    with open(file, mode='rb') as file:
        Subgraph['nnodes'] = struct.unpack('i', file.read(4))[0]
        Subgraph['nlabels'] = struct.unpack('i', file.read(4))[0]
        Subgraph['nfeats'] = struct.unpack('i', file.read(4))[0]
        Subgraph['node'] = []
        for i in range(0,Subgraph['nnodes']):
            node = {}
            node['position'] = struct.unpack('i', file.read(4))[0]
            node['truelabel'] = struct.unpack('i', file.read(4))[0]
            node['feat'] = []
            for j in range(0,Subgraph['nfeats']):
                node['feat'].append(struct.unpack('f', file.read(4))[0])
            Subgraph['node'].append(node)
    return(Subgraph)

def opf_ReadModelFile(file):
    Subgraph = {}
    with open(file, mode='rb') as file:
        Subgraph['nnodes'] = struct.unpack('i', file.read(4))[0]
        Subgraph['nlabels'] = struct.unpack('i', file.read(4))[0]
        Subgraph['nfeats'] = struct.unpack('i', file.read(4))[0]
        Subgraph['df'] = struct.unpack('f', file.read(4))[0]
        Subgraph['bestk'] = struct.unpack('i', file.read(4))[0]
        Subgraph['K'] = struct.unpack('f', file.read(4))[0]
        Subgraph['mindens'] = struct.unpack('f', file.read(4))[0]
        Subgraph['maxdens'] = struct.unpack('f', file.read(4))[0]
        Subgraph['node'] = []
        for i in range(0,Subgraph['nnodes']):
            node = {}
            node['position'] = struct.unpack('i', file.read(4))[0]
            node['truelabel'] = struct.unpack('i', file.read(4))[0]
            node['pred'] = struct.unpack('i', file.read(4))[0]
            node['label'] = struct.unpack('i', file.read(4))[0]
            node['pathval'] = struct.unpack('f', file.read(4))[0]
            node['radius'] = struct.unpack('f', file.read(4))[0]
            node['dens'] = struct.unpack('f', file.read(4))[0]
            node['feat'] = []
            for j in range(0,Subgraph['nfeats']):
                node['feat'].append(struct.unpack('f', file.read(4))[0])
            Subgraph['node'].append(node)
        Subgraph['ordered_list_of_nodes'] = []
        for i in range(0,Subgraph['nnodes']):
            Subgraph['ordered_list_of_nodes'].append(struct.unpack('i', file.read(4))[0])
    return(Subgraph)

def opf_EuclDistLog(f1, f2, n):
  return(100000.0 * math.log(opf_EuclDist(f1, f2, n) + 1))

def opf_EuclDist(f1, f2, n):
    dist = 0.0
    for i in range(0,n):
        dist += (f1[i] - f2[i]) * (f1[i] - f2[i]);
    return(dist)

def opf_OPFClassifying(sgtrain, sg):
    for i in range(0,sg['nnodes']):
        j = 0
        k = sgtrain['ordered_list_of_nodes'][j]
        #if (!opf_PrecomputedDistance)
        weight = opf_EuclDist(sgtrain['node'][k]['feat'], sg['node'][i]['feat'], sg['nfeats']);
        #else
        #weight = opf_DistanceValue(sgtrain['node'][k]['position'],sg['node'][i]['position'],sg['nfeats'])
    
        minCost = max(sgtrain['node'][k]['pathval'], weight)
        label = sgtrain['node'][k]['label']
        pred = k
        
        while((j < sgtrain['nnodes'] - 1) and (minCost > sgtrain['node'][sgtrain['ordered_list_of_nodes'][j + 1]]['pathval'])):
          l = sgtrain['ordered_list_of_nodes'][j + 1];
    
          #if (!opf_PrecomputedDistance)
          weight = opf_EuclDist(sgtrain['node'][l]['feat'], sg['node'][i]['feat'], sg['nfeats']);
          #else
          #weight = opf_DistanceValue(sgtrain['node'][l]['position'],sg['node'][i]['position'],sg['nfeats']);
          tmp = max(sgtrain['node'][l]['pathval'], weight);
          if (tmp < minCost):
            minCost = tmp;
            label = sgtrain['node'][l]['label'];
            pred = l
          j += 1;
          k = l;
        sg['node'][i]['label'] = label
        sg['node'][i]['pred'] = pred
    return(sg)

test = ReadSubgraph("dat/testing.dat")
classifier = opf_ReadModelFile("dat/classifier.opf")   
test = opf_OPFClassifying(classifier,test)  

for i in range(0,test['nnodes']):
    plt.plot(test['node'][i]['feat'][0], test['node'][i]['feat'][1], 'r^' if test['node'][i]['truelabel'] == 1 else ('ro' if test['node'][i]['truelabel'] == 2 else 'r+'))

for i in range(0,classifier['nnodes']):
    plt.plot(classifier['node'][i]['feat'][0], classifier['node'][i]['feat'][1], 'b^' if classifier['node'][i]['truelabel'] == 1 else ('bo' if classifier['node'][i]['truelabel'] == 2 else 'b+'))

for node in classifier['ordered_list_of_nodes']:
    if classifier['node'][node]['pred'] != -1:
        predFeats = classifier['node'][classifier['node'][node]['pred']]['feat']
        nodeFeats = classifier['node'][node]['feat']
        plt.arrow(predFeats[0],predFeats[1],nodeFeats[0]-predFeats[0],nodeFeats[1]-predFeats[1], length_includes_head = True, head_width = 0.15)

aux = 0
for node in test['node']:
    if node['label'] != node['truelabel']:
        aux += 1
    if node['pred'] != -1:
        predFeats = classifier['node'][node['pred']]['feat']
        nodeFeats = node['feat']
        plt.arrow(predFeats[0],predFeats[1],nodeFeats[0]-predFeats[0],nodeFeats[1]-predFeats[1], length_includes_head = True, head_width = 0.15)

print("erros:"+str(aux)+" de "+str(test['nnodes']))

G = nx.Graph()
