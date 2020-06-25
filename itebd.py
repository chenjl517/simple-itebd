# -*- coding:utf-8 -*-

import numpy as np

def itebdForCalculateHessenbergModelGroundStateEnergy(chi,T,deltaT=0.01):

    loopTimes=int(T/deltaT)

    Gama=np.random.rand(2,chi,2,chi)
    Lambda=np.random.rand(2,chi)

    H = np.array([[0.25,0,0,0],
                 [0,-0.25,0.5,0],
                 [0,0.5,-0.25,0],
                 [0,0,0,0.25]])

    w,v = np.linalg.eig(H)
    U=np.dot(np.dot(v,np.diag(np.exp(-deltaT*(w)))),np.transpose(v)) #U=e^{tH}=vwv^\dagger
    U=np.reshape(U,(2,2,2,2))

    E=0
    for i in range(loopTimes):
        A=np.mod(i,2)
        B=np.mod(i+1,2)

        #construct the tensor network
        Theta=np.tensordot(np.diag(Lambda[B,:]),Gama[A,...],axes=(1,0))
        Theta=np.tensordot(Theta,np.diag(Lambda[A,:]),axes=(2,0))
        Theta=np.tensordot(Theta,Gama[B,...],axes=(2,0))
        Theta=np.tensordot(Theta,np.diag(Lambda[B,:]),axes=(3,0))

        #apply U,contract the tensor network into a single tensor \Theta_{\alpha i j \gamma}
        Theta=np.tensordot(Theta,U,axes=((1,2),(0,1)))

        #svd
        Theta=np.reshape(np.transpose(Theta,(2,0,3,1)),(chi*2,2*chi))#chi,d,d,chi
        X,newLambda,Y=np.linalg.svd(Theta)
        # print(newLambda)

        #contract the Lambda: Truncate the lambda and renormalization
        Lambda[A,:]=newLambda[0:chi]/np.sqrt(np.sum(newLambda[0:chi]**2))
        # print(np.sum(Lambda[A,]**2))

        #construct X,introduce lambda^B back into the network
        X=X[0:2*chi,0:chi]
        X=np.reshape(X,(2,chi,chi))
        X=np.transpose(X,(1,0,2))
        Gama[A,...]=np.tensordot(np.diag(Lambda[B,:]**(-1)),X,axes=(1,0))

        #construct Y,introduce lambda^B back into the network
        Y=Y[0:chi,0:2*chi]
        Y=np.reshape(Y,(chi,2,chi))
        Gama[B,...]=np.tensordot(Y,np.diag(Lambda[B,:]**(-1)),axes=(2,0))

        if(i>=loopTimes-2):
            E+=-np.log(np.sum(Theta**2))/(deltaT*2)
            # print("loop times:",i,"E =", -np.log(np.sum(Theta**2))/(deltaT*2))

    # print("E=",E/2)
    return E/2