
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random
import tensorflow as tf
from tqdm import tqdm_notebook
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from keras.utils import np_utils


# In[2]:


humanObservedData = pd.read_csv('HumanObserved-Dataset\HumanObserved-Features-Data\HumanObserved-Features-Data.csv')
s_HumanObservedData = humanObservedData.drop(humanObservedData.columns[0], axis=1).reset_index(drop=True)
formattedHumanObservedData = s_HumanObservedData.set_index('img_id')


# In[3]:


humanObservedData_DifferentPairs = pd.read_csv('HumanObserved-Dataset\HumanObserved-Features-Data\diffn_pairs.csv')


# In[4]:


humanObservedData_SamePairs = pd.read_csv('HumanObserved-Dataset\HumanObserved-Features-Data\same_pairs.csv')


# In[5]:


concatenatedData = []
subtractedData = []
for index, row in humanObservedData_SamePairs.iterrows():
    img_A = row['img_id_A']
    img_B = row['img_id_B']
    target = row['target']
    x = formattedHumanObservedData.loc[img_A].values.tolist()
    y = formattedHumanObservedData.loc[img_B].values.tolist()
    z = [i-j for i,j in zip(x,y)]
#     z.insert(0, img_B)
#     z.insert(0, img_A)
    z.append(target)
    subtractedData.append(z)
    x.extend(y)
    x.append(target)
#     x.insert(0, img_B)
#     x.insert(0, img_A)
    concatenatedData.append(x)

humanObservedData_SamePairs_Size = len(humanObservedData_SamePairs)
sampledHumanObservedData_DifferentPairs = pd.DataFrame()
if(len(humanObservedData_DifferentPairs) > len(humanObservedData_SamePairs)):
    sampledHumanObservedData_DifferentPairs = humanObservedData_DifferentPairs.sample(humanObservedData_SamePairs_Size)
else:
    sampledHumanObservedData_DifferentPairs = humanObservedData_DifferentPairs

for index, row in sampledHumanObservedData_DifferentPairs.iterrows():
    img_A = row['img_id_A']
    img_B = row['img_id_B']
    target = row['target']
    x = formattedHumanObservedData.loc[img_A].values.tolist()
    y = formattedHumanObservedData.loc[img_B].values.tolist()
    z = [i-j for i,j in zip(x,y)]
#     z.insert(0, img_B)
#     z.insert(0, img_A)
    z.append(target)
    subtractedData.append(z)
    x.extend(y)
    x.append(target)
#     x.insert(0, img_B)
#     x.insert(0, img_A)
    concatenatedData.append(x)

concatenatedHumanObservedFeatureSet = pd.DataFrame(concatenatedData, columns=['fA1','fA2','fA3','fA4','fA5','fA6','fA7','fA8','fA9','fB1','fB2','fB3','fB4','fB5','fB6','fB7','fB8','fB9','t'])
subtractedHumanObservedFeatureSet = pd.DataFrame(subtractedData,columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','t'])


# In[6]:


GSCData = pd.read_csv('GSC-Dataset\GSC-Features-Data\GSC-Features.csv')
formattedGSCData = GSCData.set_index('img_id')


# In[7]:


GSCData_DifferentPairs = pd.read_csv('GSC-Dataset\GSC-Features-Data\diffn_pairs.csv')


# In[90]:


GSCData_SamePairs = pd.read_csv('GSC-Dataset\GSC-Features-Data\same_pairs.csv')
GSCData_SamePairs = GSCData_SamePairs.sample(1000)


# In[91]:


concatenatedData = []
subtractedData = []
for index, row in GSCData_SamePairs.iterrows():
    img_A = row['img_id_A']
    img_B = row['img_id_B']
    target = row['target']
    x = formattedGSCData.loc[img_A].values.tolist()
    y = formattedGSCData.loc[img_B].values.tolist()
    z = [i-j for i,j in zip(x,y)]
#     z.insert(0, img_B)
#     z.insert(0, img_A)
    z.append(target)
    subtractedData.append(z)
    x.extend(y)
    x.append(target)
#     x.insert(0, img_B)
#     x.insert(0, img_A)
    concatenatedData.append(x)

sampledGSCData_DifferentPairs = GSCData_DifferentPairs.sample(len(GSCData_SamePairs))

for index, row in sampledGSCData_DifferentPairs.iterrows():
    img_A = row['img_id_A']
    img_B = row['img_id_B']
    target = row['target']
    x = formattedGSCData.loc[img_A].values.tolist()
    y = formattedGSCData.loc[img_B].values.tolist()
    z = [i-j for i,j in zip(x,y)]
#     z.insert(0, img_B)
#     z.insert(0, img_A)
    z.append(target)
    subtractedData.append(z)
    x.extend(y)
    x.append(target)
#     x.insert(0, img_B)
#     x.insert(0, img_A)
    concatenatedData.append(x)

concatenatedGSCFeatureSet = pd.DataFrame(concatenatedData)
subtractedGSCFeatureSet = pd.DataFrame(subtractedData)


# In[92]:


#Training Percentage
TrainingPercent = 80
#Validation Percentage
ValidationPercent = 10
#Test Percentage
TestingPercent = 10
# Number of clusters
PHI = []


# In[93]:


def Sigmoid(A_n):
    return 1.0 / (1 + np.exp(A_n))


# In[94]:


def GenerateTrainingData(RawFeatureSet, TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(RawFeatureSet)*(TrainingPercent*0.01)))
    t = RawFeatureSet.iloc[:TrainingLen]
    return t


# In[95]:


def GenerateTestingData(RawFeatureSet, TrainingCount, TestingPercent = 10):
    TestingLen = int(math.ceil(len(RawFeatureSet)*(TestingPercent*0.01)))
    TestingEnd = TrainingCount + TestingLen
    t = RawFeatureSet.iloc[:TestingEnd]
    return t


# In[96]:


def GetLinearValTest(FeatureSet,W):
    Y = np.dot(W,np.transpose(FeatureSet))
    return Y


# In[97]:


def GetLogisticValTest(FeatureSet,W):
    Y = Sigmoid(np.dot(W,np.transpose(FeatureSet)))
    return Y


# In[98]:


def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# In[99]:


def GetAccuracy(RegressionTarget, ExistingTarget):
    accuracy = 0.0
    ActualTarget = ExistingTarget.values
    RegressionTargetOutput = [-1 for i in range(0, len(RegressionTarget))]
    if(len(RegressionTarget) == len(ActualTarget)):
        matchedTargets = 0
        for i in range(0,len(RegressionTarget)):
            if(RegressionTarget[i] >= 0.5):
                RegressionTarget[i] = 1
            else:
                RegressionTarget[i] = 0
        for i in range(0,len(RegressionTarget)):
            if(RegressionTarget[i] == ActualTarget[i]):
                matchedTargets += 1
        accuracy = (matchedTargets/len(ActualTarget))*100
    return accuracy


# In[100]:


def GenerateBigSigma(Data, MuMatrix, TrainingPercent):
#     Data = Data.values
    # The entire data is transposed.
    DataT       = np.transpose(Data)
    # Initialize the Big Sigma matrix of zeros
    BigSigma    = np.zeros((len(DataT),len(DataT)))

    # Training length is calculated based on the training percent and transposed data.
    TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))

    varVect     = []

    # Variance is calculated for each row in the data and put into the vector.
    for i in range(0,len(Data.iloc[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(DataT.iloc[i,j])    
        varVect.append(np.var(vct))

    # Diagonal matrix containing the variance of the dataset is generated.
    for j in range(len(DataT)):
        BigSigma[j][j] = varVect[j] + 0.2

    BigSigma = np.dot(0.001,BigSigma)

    return BigSigma


# In[101]:


def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x


# In[102]:


def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    Data = Data.values
    TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(Data[R], MuMatrix[C], BigSigInv)
    return PHI


# In[103]:


def NormalizeFeatureSet(FeatureSet):
    TargetColumnLocation = len(FeatureSet.columns) - 1
    FeatureTarget = FeatureSet.iloc[:,TargetColumnLocation]
    JustFeatureSet = FeatureSet.iloc[:,:TargetColumnLocation]
#     NormalizedFeatureSet = (JustFeatureSet - JustFeatureSet.values.min())/(JustFeatureSet.values.max()-JustFeatureSet.values.min())
#     return NormalizedFeatureSet, FeatureTarget
    return JustFeatureSet, FeatureTarget


# In[104]:


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))


# In[105]:


def ApplyLinearRegression(FeatureSet, M = 3, Lambda = 1, LearningRate = 0.01, NumberOfSamples = 500):
    NormalizedFeatureSet, FeatureTarget = NormalizeFeatureSet(FeatureSet)

    TrainingData = GenerateTrainingData(NormalizedFeatureSet, TrainingPercent)
    TrainingTarget = GenerateTrainingData(FeatureTarget, TrainingPercent)

    TestingData = GenerateTestingData(NormalizedFeatureSet, len(TrainingData), TestingPercent)
    TestingTarget = GenerateTestingData(FeatureTarget, len(TrainingTarget), TestingPercent)

    ValidationData = GenerateTestingData(NormalizedFeatureSet, len(TestingData), ValidationPercent)
    ValidationTarget = GenerateTestingData(FeatureTarget, len(TestingTarget), ValidationPercent)

    kmeans = KMeans(n_clusters=M, random_state=0).fit(TrainingData)

    Mu = kmeans.cluster_centers_

    BigSigma = GenerateBigSigma(NormalizedFeatureSet, Mu, TrainingPercent)

    Training_Phi = GetPhiMatrix(NormalizedFeatureSet, Mu, BigSigma, TrainingPercent)

    Validation_Phi = GetPhiMatrix(ValidationData, Mu, BigSigma, 100)

    Testing_Phi = GetPhiMatrix(TestingData, Mu, BigSigma, 100)
    
    W = [1 for i in range(len(Training_Phi[0]))]
    W = np.dot(random.uniform(-0.2,0.2), W)
    
    W_Now        = W
    #Lambda
    La           = Lambda
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    L_Accuracy_TR = []
    L_Accuracy_Val = []
    L_Accuracy_Test = []
    
    for i in range(0,NumberOfSamples):      
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),Training_Phi[i])),Training_Phi[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(LearningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        TR_TEST_OUT   = GetLinearValTest(Training_Phi,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        L_Accuracy_TR.append(float(Erms_TR.split(',')[0]))
        
        VAL_TEST_OUT  = GetLinearValTest(Validation_Phi,W_T_Next)
        Erms_Val      = GetErms(VAL_TEST_OUT,ValidationTarget)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        L_Accuracy_Val.append(float(Erms_Val.split(',')[0]))
        
        TEST_OUT      = GetLinearValTest(Testing_Phi,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestingTarget)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        L_Accuracy_Test.append(float(Erms_Test.split(',')[0]))
        
    print ("-------------------------- LINEAR REGRESSION ------------------------------")
    print("Lambda = "+str(La)+" Learning Rate = "+str(LearningRate))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ("Accuracy Training   = " + str(np.around(L_Accuracy_TR[len(L_Accuracy_TR)-1],5)))
    print ("Accuracy Validation = " + str(np.around(L_Accuracy_Val[len(L_Accuracy_Val)-1],5)))
    print ("Accuracy Testing    = " + str(np.around(L_Accuracy_Test[len(L_Accuracy_Test)-1],5)))


# In[106]:


def ApplyLogisticRegression(FeatureSet, M=3, LearningRate=0.01, NumberOfSamples = 500):
    NormalizedFeatureSet, FeatureTarget = NormalizeFeatureSet(FeatureSet)

    TrainingData = GenerateTrainingData(NormalizedFeatureSet, TrainingPercent)
    TrainingTarget = GenerateTrainingData(FeatureTarget, TrainingPercent)

    TestingData = GenerateTestingData(NormalizedFeatureSet, len(TrainingData), TestingPercent)
    TestingTarget = GenerateTestingData(FeatureTarget, len(TrainingTarget), TestingPercent)

    ValidationData = GenerateTestingData(NormalizedFeatureSet, len(TestingData), ValidationPercent)
    ValidationTarget = GenerateTestingData(FeatureTarget, len(TestingTarget), ValidationPercent)

    kmeans = KMeans(n_clusters=M, random_state=0).fit(TrainingData)

    Mu = kmeans.cluster_centers_

    BigSigma = GenerateBigSigma(NormalizedFeatureSet, Mu, TrainingPercent)

    Training_Phi = GetPhiMatrix(NormalizedFeatureSet, Mu, BigSigma, TrainingPercent)

    Validation_Phi = GetPhiMatrix(ValidationData, Mu, BigSigma, 100)

    Testing_Phi = GetPhiMatrix(TestingData, Mu, BigSigma, 100)
         
    W = [1 for i in range(len(Training_Phi[0]))]
    W = np.dot(random.uniform(-0.2,0.2), W)
    
    W_Now        = W
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    L_Accuracy_TR = []
    L_Accuracy_Val = []
    L_Accuracy_Test = []
    
    
    for i in range(0,NumberOfSamples):
        
        Delta_E      = np.dot((Sigmoid(np.dot(np.transpose(W_Now),Training_Phi[i])) - TrainingTarget[i]),Training_Phi[i])  
        Delta_W      = -np.dot(LearningRate,Delta_E)
        W_T_Next     = W_Now + Delta_W
        W_Now        = W_T_Next

        TR_TEST_OUT   = GetLogisticValTest(Training_Phi,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Accuracy_TR.append(GetAccuracy(TR_TEST_OUT,TrainingTarget))
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        VAL_TEST_OUT  = GetLogisticValTest(Validation_Phi,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,ValidationTarget)
        L_Accuracy_Val.append(GetAccuracy(VAL_TEST_OUT,ValidationTarget))
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        TEST_OUT      = GetLogisticValTest(Testing_Phi,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestingTarget)
        L_Accuracy_Test.append(GetAccuracy(TEST_OUT,TestingTarget))
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    
    print ("-------------------------- LOGISTIC REGRESSION ------------------------------")
    print("Learning Rate = "+str(LearningRate))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ("Accuracy Training   = " + str(np.around(L_Accuracy_TR[len(L_Accuracy_TR)-1],5)))
    print ("Accuracy Validation = " + str(np.around(L_Accuracy_Val[len(L_Accuracy_Val)-1],5)))
    print ("Accuracy Testing    = " + str(np.around(L_Accuracy_Test[len(L_Accuracy_Test)-1],5)))


# In[109]:


def ApplyNeuralNetwork(FeatureSet, NumberOfNeurons = 512, LearningRate = 0.01, NumberOfEpochs = 1000, BatchSize = 128):
    NormalizedFeatureSet, FeatureTarget = NormalizeFeatureSet(FeatureSet)

    TrainingData = GenerateTrainingData(NormalizedFeatureSet, TrainingPercent)
    TrainingTarget = GenerateTrainingData(FeatureTarget, TrainingPercent)

    TestingData = GenerateTestingData(NormalizedFeatureSet, len(TrainingData), TestingPercent)
    TestingTarget = GenerateTestingData(FeatureTarget, len(TrainingTarget), TestingPercent)

    ValidationData = GenerateTestingData(NormalizedFeatureSet, len(TestingData), ValidationPercent)
    ValidationTarget = GenerateTestingData(FeatureTarget, len(TestingTarget), ValidationPercent)
    
    TrainingData = TrainingData.values
    TrainingTarget = np_utils.to_categorical(np.array(TrainingTarget.values),2)

    ValidationData = ValidationData.values
    ValidationTarget = np_utils.to_categorical(np.array(ValidationTarget.values),2)

    TestingData = TestingData.values
    TestingTarget = np_utils.to_categorical(np.array(TestingTarget.values),2)

    inputTensor  = tf.placeholder(tf.float32, [None, TrainingData.shape[1]])
    outputTensor = tf.placeholder(tf.float32, [None, 2])

    NUM_HIDDEN_NEURONS_LAYER_1 = NumberOfNeurons
    LEARNING_RATE = LearningRate

    input_hidden_weights  = init_weights([TrainingData.shape[1], NUM_HIDDEN_NEURONS_LAYER_1])

    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 2])

    hidden_layer = tf.nn.sigmoid(tf.matmul(inputTensor, input_hidden_weights))

    output_layer = tf.matmul(hidden_layer, hidden_output_weights)

    error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

    training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

    prediction = tf.argmax(output_layer, 1)

    NUM_OF_EPOCHS = NumberOfEpochs
    BATCH_SIZE = BatchSize

    training_accuracy = []

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):

            p = np.random.permutation(range(len(TrainingData)))
            TrainingData  = TrainingData[p]
            TrainingTarget = TrainingTarget[p]

            for start in range(0, len(TrainingData), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(training, feed_dict={inputTensor: TrainingData[start:end], 
                                              outputTensor: TrainingTarget[start:end]})

            training_accuracy.append(np.mean(np.argmax(TrainingTarget, axis=1) ==
                                 sess.run(prediction, feed_dict={inputTensor: TrainingData,
                                                                 outputTensor: TrainingTarget})))

        predictedValidationTarget = sess.run(prediction, feed_dict={inputTensor: ValidationData})

        predictedTestTarget = sess.run(prediction, feed_dict={inputTensor: TestingData})

    wrong   = 0
    right   = 0

    predictedValidationTargetList = []

    for i,j in zip(ValidationTarget,predictedValidationTarget):
        predictedValidationTargetList.append(j)

        if np.argmax(i) == j:
            right = right + 1
        else:
            wrong = wrong + 1
    
    print ("-------------------------- NEURAL NETWORK ------------------------------")
    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("Training Accuracy: " + str(right/(right+wrong)*100)) 

    wrong   = 0
    right   = 0

    predictedTestTargetList = []

    for i,j in zip(TestingTarget,predictedTestTarget):
        predictedTestTargetList.append(j)

        if np.argmax(i) == j:
            right = right + 1
        else:
            wrong = wrong + 1

    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("Testing Accuracy: " + str(right/(right+wrong)*100))


# In[43]:


print("---------- Human Observed Dataset with feature concatentation -------------")
ApplyLinearRegression(concatenatedHumanObservedFeatureSet.sample(frac=1).reset_index(drop=True),Lambda=0.1,M=4,LearningRate=0.01,NumberOfSamples=1000)
ApplyLogisticRegression(concatenatedHumanObservedFeatureSet.sample(frac=1).reset_index(drop=True),LearningRate=0.01,M=4,NumberOfSamples=1000)
ApplyNeuralNetwork(concatenatedHumanObservedFeatureSet.sample(frac=1).reset_index(drop=True), NumberOfNeurons=len(concatenatedHumanObservedFeatureSet),BatchSize=256,LearningRate=0.001,NumberOfEpochs=2000)


# In[64]:


print("---------- Human Observed Dataset with feature subtraction -------------")
ApplyLinearRegression(subtractedHumanObservedFeatureSet.sample(frac=1).reset_index(drop=True),Lambda=1,M=3,LearningRate=0.1,NumberOfSamples=1000)
ApplyLogisticRegression(subtractedHumanObservedFeatureSet.sample(frac=1).reset_index(drop=True),LearningRate=0.001,M=3,NumberOfSamples=1000)
ApplyNeuralNetwork(subtractedHumanObservedFeatureSet.sample(frac=1).reset_index(drop=True), NumberOfNeurons=len(subtractedHumanObservedFeatureSet),BatchSize=15,LearningRate=0.001,NumberOfEpochs=1000)


# In[88]:


print("---------- GSC Dataset with feature concatentation -------------")
ApplyLinearRegression(concatenatedGSCFeatureSet.sample(frac=1).reset_index(drop=True))
ApplyLogisticRegression(concatenatedGSCFeatureSet.sample(frac=1).reset_index(drop=True))
ApplyNeuralNetwork(concatenatedGSCFeatureSet.sample(frac=1).reset_index(drop=True))


# In[108]:


print("---------- GSC Dataset with feature subtraction -------------")
ApplyLinearRegression(subtractedGSCFeatureSet.sample(frac=1).reset_index(drop=True))
ApplyLogisticRegression(subtractedGSCFeatureSet.sample(frac=1).reset_index(drop=True))
ApplyNeuralNetwork(subtractedGSCFeatureSet.sample(frac=1).reset_index(drop=True))

