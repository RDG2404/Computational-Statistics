import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM as gHMM
import os
# import matplotlib.pyplot 
import matplotlib.pyplot as plt
####
def predict_next_day(model,currData):
    # nStates = model.n_components
    # mean = []
    # stddev = []
    futData = []
    futData = np.linspace(0.9*float(min(currData)),1.1*float(max(currData)),50)
    '''
    for k in range(nStates):
        mean.append(model.means_[k])
        stddev.append(np.sqrt(model.covars_[k]))
        futData = np.append(futData,
                np.linspace(mean[k]-2*stddev[k],mean[k]+2*stddev[k],20))
    '''
    # breakpoint()
    logLik = []
    for data in futData:
        reqData = np.append(currData,data)[:,np.newaxis]
        logLik.append(model.score(reqData))
    # breakpoint()
    obs_pred = futData[np.argmax(logLik)]    
    # hidden_states = model.decode(np.append(currData,obs_pred))
    # breakpoint()
    return obs_pred #,hidden_states[-1]


#####

os.chdir('D:/ChakriBackup/GATech_MSA/Spring2022/ISYE6416/LitSurvey_Project')
rawdata = pd.read_csv('CL_NMX_CrudeNasdaq.csv')
data = rawdata[::-1].reset_index(drop = True)

data.rename(columns = {'Close/Last':'Close'}, inplace = True)

data["CloseRat"] = 0
data["HighRat"] = 0
data["LowRat"] = 0

for ind in range(1,len(data)):
    data["CloseRat"].iloc[ind] = 100*(data["Close"].iloc[ind]-data["Close"].iloc[ind-1])/data["Close"].iloc[ind-1]
    data["HighRat"].iloc[ind] = 100*(data["High"].iloc[ind]-data["Close"].iloc[ind-1])/data["Close"].iloc[ind-1]
    data["LowRat"].iloc[ind] = 100*(data["Low"].iloc[ind]-data["Close"].iloc[ind-1])/data["Close"].iloc[ind-1]

# data.drop(columns=['Close','Open','High','Low','Volume'], inplace = True)
# breakpoint()
data.drop([0],inplace = True)
# breakpoint()
window = 30
# model = gHMM(n_components=2,covariance_type="diag")
nHidden = 2
nIter = 100
# model = gHMM(n_components=nHidden,covariance_type="spherical",n_iter=nIter)
# model.monitor_.verbose = True

predPrice = []
meansData = []
covData = []
truePrice = []
hiddenState = []
modifyTransMat = True
troublePoint = False
probCases = 0
# predPriceDim = []
# truePriceDim = []
for ind in range(window,len(data)-1):
    reqData = data.iloc[ind-window:ind,:]
    # breakpoint()
    reqData = np.array(reqData["Close"])[:,np.newaxis]
    model = gHMM(n_components=nHidden,covariance_type="diag",n_iter=nIter)
    # model.means_prior = np.array([[1.3],[0.1],[-1.5]])
    # model.means_ = np.array([[2.5],[1.2],[0.0],[-1.2],[-2.5]])
    # model.params = 'stc'
    model.covars_prior = 0.3
    # model.init_params = 'stc'
    model.min_covar= 0.1
    troublePoint = False
    model.fit(reqData)
    transSum = np.sum(model.transmat_,axis=1)
    for rowInd in range(len(transSum)):
        if(transSum[rowInd] == 0 and modifyTransMat):
           # model.transmat_[rowInd] = 0.999
           # model.transmat_[rowInd,rowInd] = 0.001
           probCases = probCases + 1
           troublePoint = True
           # breakpoint()
            
    # print(np.round(model.transmat_,3))
    if(troublePoint == True):
        obs_pred = 0.0
    else:
        obs_pred = predict_next_day(model,reqData)
    predObsSeq = []
    predStateSeq = []
    predObsSeq = np.append(reqData,obs_pred)[:,np.newaxis]
    if(troublePoint == True):
        model.fit(predObsSeq)
        predStateSeq = model.decode(predObsSeq)
    else:
        predStateSeq = model.decode(predObsSeq)
    predPrice.append(obs_pred)
    truePrice.append(data["Close"][ind+1])
    # truePrice.append(data.iloc[ind,1])
    # breakpoint()
    # Hidden state processing
    tempList = []
    temp2 = []
    tempList = np.concatenate(
        (model.means_,np.linspace(0,nHidden-1,nHidden)[:,None],model.covars_[:,0]),
        axis = 1)
    tempList = pd.DataFrame(tempList,columns=['mean','state','vars'])
    tempList.sort_values(by=['mean'],inplace=True)
    temp2 = np.where(tempList['state']==predStateSeq[-1][-1])
    meansData.append(np.array(tempList['mean']))
    hiddenState.append(temp2[0][0])
    covData.append(np.array(tempList['vars']))
    
    # print(ind)
    # breakpoint()
    # hiddenState.append(predStateSeq[-1][-1])
    
    # predPriceDim.append(data["Close"][ind]*(1+0.01*obs_pred))
    # truePriceDim.append(data["Close"][ind+1])
    # breakpoint()
    # print(model.transmat_prior)
    
fig = plt.figure()
plt.plot(predPrice[-100:],'r',truePrice[-100:],'b')
plt.ylim(-5,5)
plt.show()

print(np.corrcoef(truePrice,predPrice))

fig = plt.figure()
plt.plot(predPrice[-100:],'r',truePrice[-100:],'b')
# plt.ylim(-5,5)
plt.show()
# print(np.corrcoef(truePrice,predPrice))

fig, axs = plt.subplots(3)
# fig.suptitle('Vertically stacked subplots')
axs[0].plot(predPrice[-600:-500],'r',truePrice[-600:-500],'b')
axs[1].plot(hiddenState[-600:-500],'r')
axs[2].plot(np.array(meansData)[-600:-500])
axs[2].set_ylim([25,75])
# axs[2].ylim(-10, 10)

# fig = plt.figure()
# plt.plot(np.array(meansData)[-600:-500])
# plt.ylim(-20, 20)
# plt.show()

# ylim(-5,5)
'''
breakpoint()

model.fit(np.array(reqData.iloc[:,1])[:,np.newaxis])
print(model.means_)
print(model.transmat_)

print(model.transmat_)
reqData = reqData.iloc[:,1]
reqData = np.append(reqData,[2.3])
reqData = np.array(reqData)[:,np.newaxis]
model.score(reqData)
model.decode(reqData)

model2 = gHMM(n_components=2,covariance_type="spherical")
model2.fit(reqData)
model2.decode(reqData)
'''
