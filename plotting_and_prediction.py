from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time

dat = 2000
total = 1950
train = 1700
test = 250
pr = 3

inputFD = np.load('/content/drive/MyDrive/FEM/LHS(' + str(dat) + ')_inputFD.npy')
# inputTD = np.load('/content/drive/MyDrive/FEM/LHS(' + str(dat) + ')_inputTD.npy')
output = np.load('/content/drive/MyDrive/FEM/LHS(' + str(dat) + ')_output.npy')

"SCALING DATA"

scaler1 = MinMaxScaler((0, 1))
scaler2 = MinMaxScaler((0, 1))
scaler = MinMaxScaler((-1, 1))

inputFD = scaler.fit_transform(inputFD)
# inputTD = scaler.fit_transform(inputTD)
output = np.squeeze(scaler1.fit_transform(np.expand_dims(output[:, pr], axis=1)), axis=1)

"CREATING TENSORS"

# X_TD = torch.from_numpy(inputTD[0:dat,:].astype(np.float32))
# y_TD = torch.from_numpy(output[0:dat].astype(np.float32))
X_FD = torch.from_numpy(inputFD[0:dat, :].astype(np.float32))
y_FD = torch.from_numpy(output[0:dat].astype(np.float32))

"TRAINING WITH JUST FD RESPONSE"

print("ONE LAYER RBF")

n_clusters=300
n_neighbors=4
kmeans = KMeans(n_clusters, random_state=0).fit(inputFD[0:train,:])
c=kmeans.cluster_centers_
nbrs = NearestNeighbors(n_neighbors+1, algorithm='ball_tree').fit(kmeans.cluster_centers_)
distances, indices = nbrs.kneighbors(c)
distances=np.delete(distances, 0, 1)

x=np.zeros((n_clusters))
for i in range(n_clusters):
  for j in range(n_neighbors):
    x[i]+=(distances[i][j])**2
r=np.sqrt(x/(n_neighbors))

rbf1 = ONE_LAYER_RBF_NN(n_clusters, 27, gaussian)
tic = time.time()
rbf1.fit(X_FD[0:total], y_FD[0:total], 100, 10, 0.0001, nn.MSELoss(), train, test)
toc = time.time()
print("Time taken: ", (toc-tic)*1000, "ms")

error_size=50
np.set_printoptions(precision=2)
print("ACTUAL VALUES")
act1 = scaler1.inverse_transform(np.expand_dims(output[total:total+error_size], axis=1)).T
print(act1[:,0:20])

print("PREDICTED VALUES OF TWO LAYER RBF")
y_pred1 = rbf1(torch.from_numpy(inputFD[total:total+error_size,:].astype(np.float32)))
Y_pred1 = scaler1.inverse_transform(y_pred1.detach().numpy()).T
print(Y_pred1[:,0:10])
rmse_1 = np.sqrt(np.sum((act1-Y_pred1).squeeze(0)**2))/error_size
print("rmse error",rmse_1)
perct_1 = (np.sum(abs((act1-Y_pred1)/act1)))*(100/(np.shape(act1)[1]))
print("prcnt error", perct_1)
max_1 = 100*np.amax(abs((act1-Y_pred1)/act1))
print("max prcnt error", max_1)

print("TWO LAYER MLP")
mlp2 = TWO_LAYER_MLP(27, 20, 10)
tic = time.time()
mlp2.fit(X_FD[0:total], y_FD[0:total], 500, 1000, 0.001, nn.MSELoss(), train, test)
toc = time.time()
print("Time taken: ", (toc-tic)*1000, "ms")

error_size=50
np.set_printoptions(precision=2)
print("ACTUAL VALUES")
act1 = scaler1.inverse_transform(np.expand_dims(output[total:total+error_size], axis=1)).T
print(act1[:,0:20])

print("PREDICTED VALUES OF THREE LAYER MLP")
y_pred4 = mlp2(torch.from_numpy(inputFD[total:total+error_size,:].astype(np.float32)))
Y_pred4 = scaler1.inverse_transform(y_pred4.detach().numpy()).T
print(Y_pred4[:,0:20])
rmse_4 = np.sqrt(np.sum((act1-Y_pred4).squeeze(0)**2))/error_size
print("rmse error",rmse_4)
perct_4 = (np.sum(abs((act1-Y_pred4)/act1)))*(100/(np.shape(act1)[1]))
print("prcnt error", perct_4)
max_4 = 100*np.amax(abs((act1-Y_pred4)/act1))
print("max prcnt error", max_4)
