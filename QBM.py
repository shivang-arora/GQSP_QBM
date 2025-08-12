
from hamiltonian import *
from GQEVT import *
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy
from sklearn.metrics import accuracy_score
from datasets import * 


class QBM1:


    def __init__(self,n_hidden,n_visible,n_output,terms=None,connectivity='all',n=20,beta=0.1):
        
        self.hamiltonian=ModelHamiltonian(n_hidden,n_visible,terms=terms,n_output=n_output,connectivity='all')
        
        self.n=n
        self.beta=beta
        self.gqevt=GQEVT(n,-beta,'GQSP')
        #self.initial_params=initial_params
        #self.params=initial_params
        self.n_hidden=self.hamiltonian.n_hidden
        self.n_visible=self.hamiltonian.n_visible
        
        self.n_qubits=self.n_hidden+self.n_visible
        self.Ny=self.hamiltonian.Ny
        #self.n_y=1
        
        #self.n_x=n_visible-self.n_y
       

        self.weights= self.hamiltonian.get_weights()
       
    
    
    def update_model_params(self,weights):
        # reorder weights for simplicity
        
        i=0
        for key,value in self.hamiltonian.params.items():

            m=len(value)

            self.hamiltonian.params[key]=weights[i:i+m]

            i=i+1
            

        
    def _output_clamped_configs(self,x,y):
        
        
        
        self.H_x=self.hamiltonian.build_hamiltonians(x, y=None)
        self.H_xy=self.hamiltonian.build_hamiltonians(x, y)
        
    
        
        x_clamped = self._compute_expectation(self.H_x) 
        x_clamped=np.array(x_clamped)/x_clamped[0]   
            
        xy_clamped = self._compute_expectation(self.H_xy) 
        xy_clamped=np.array(xy_clamped)/xy_clamped[0]   
        
        
        
        return x_clamped,xy_clamped

        
       
        
        
            
    
    def _compute_expectation(self,H):
        
        
        
        
        new_wires=range(2,self.n_qubits+2)
        old_wires=range(0,self.n_qubits)
        
        
        H_new = H.map_wires(dict(zip(old_wires,new_wires)))
        proj = qml.Projector( [0] * 2,wires=range(0,2))
        new_ops=[proj]+[proj@op for op in H_new.ops]
        
        dev_xy = qml.device("default.qubit", wires=2*(len(H.wires))+2)
        self.gqevt.build(qml.matrix(H_new))
        @qml.qnode(dev_xy, interface="autograd")
        
        def _compute():

            
            for wire in range(2,len(H.wires)+2):
                qml.Hadamard(wire)
                qml.CNOT([wire,wire+len(H.wires)])
            
           
            self.gqevt.circuit(self.gqevt.angles,'direct_matrix')
            #return qml.expval(qml.PauliZ(0))
            #return qml.state()
            return [qml.expval(op) for op in new_ops]
            
        return _compute()    
        
        
    def update_weights(self,x_batch,y_batch,learning_rate):
        
        errors=0 
        
      
        for i,x_vector in enumerate(x_batch):
            
           
            y_vector=y_batch[i]
            y_vector=2*y_vector-1
            
           
            
            x_clamped,xy_clamped=self._output_clamped_configs(x_vector,y_vector)
            
            
            x_clamped=x_clamped[1:]
            
            xy_clamped=xy_clamped[1:]
            
            
            new_x_clamped,new_xy_clamped= self.hamiltonian.order_match_configs(x_clamped,xy_clamped,x_vector,y_vector,self.H_x.terms()[-1],self.H_xy.terms()[-1])
            
           
            
            errors+= (new_xy_clamped - new_x_clamped)

            
            
            
            
            
            

        errors /= x_batch.shape[0]
      
        self.weights = self.weights + learning_rate * errors
        #print('length of weight vector', len(self.weights))
        
        self.update_model_params(self.weights)
        
      
        
            
            
        return errors

    def predict(self,x):
        
      
        H_x=self.hamiltonian.build_hamiltonians(x, y=None)
        
    
        ops=[qml.PauliZ(i) for i in range(self.n_hidden,self.n_hidden+self.Ny)]
        
        x_clamped = self._compute_expectation(H_x) 
        
        x_clamped=np.array(x_clamped)/x_clamped[0]

        indices=[H_x.terms()[-1].index(op) for op in ops]

     
        output_vals=np.array([x_clamped[1+index] for index in indices])
        
       
        
        y_predict=(np.sign(output_vals)+1)/2
        
        prediction=0
        for i in range(len(y_predict)):
            prediction+=(2**i )* (y_predict[len(y_predict)-i-1])
        return output_vals,prediction
     
    def predict_test_new(self,X_test,X_raw):
        predictions=[] 
        for x_vector in X_test:
            
             exp_y,y_predict=self.predict(x_vector)
             predictions.append(y_predict)

       #
        visualize_samples(X_raw,predictions,range(0,12),6)
        acc_sklearn = accuracy_score(y_sel, predictions)
        print(f"Accuracy (sklearn): {acc_sklearn:.4f}")

        
       
        #plt.axline((0, 0), slope=1, linestyle='--', color='gray', label="Decision boundary: x1 + x2 = 0")
        plt.show()
    
    def predict_test(self,X_test,y_test,plot=True):
        predictions=[] 
        for x_vector in X_test:
            
             exp_y,y_predict=self.predict(x_vector)
             predictions.append(y_predict)

        def to_cat(y_test):
            predictions=[]
            for y_predict in y_test:
                 prediction=0
                 for i in range(len(y_predict)):
                    prediction+=(2**i )* (y_predict[len(y_predict)-i-1])
                 predictions.append(prediction)
            return np.array(predictions)
        
        y_cat=to_cat(y_test)
        acc_sklearn = accuracy_score(y_cat, predictions)
        print(f"Accuracy (sklearn): {acc_sklearn:.4f}")
       
        
        if plot==True:
            plt.subplot(1,2,1)
            plt.title('Predictions')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='x', label="Test")
    
            plt.subplot(1,2,2)
            plt.title('Ground truth')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=to_cat(y_test), cmap='viridis', marker='x', label="Test")
            
            #plt.axline((0, 0), slope=1, linestyle='--', color='gray', label="Decision boundary: x1 + x2 = 0")
            plt.show()
        return acc_sklearn
    
    def train_model(self, X,y_data, X_test,y_test,batch_size=8, learning_rate=0.005,epochs=3,plot=False):
        self.accuracy=[]
        self.epochs=epochs
        if  len(X[0])+len(y_data[0])!= self.n_visible:
            raise ValueError(f" Insufficient visible nodes for dataset")
        
        data=X
      
        
        batch_num = data.shape[0] // batch_size
        
        diff = data.shape[0] % batch_size
        
        self.batch_size=batch_size
        
        if diff:
            
        
            data = data[:-diff]
            y_data= y_data[:-diff]
            last_batch = data[data.shape[0] - diff:]
            last_ybatch= y_data[y_data.shape[0] - diff:]
        
      
        x_batches = np.vsplit(data, batch_num)
     
        y_batches=np.vsplit(y_data, batch_num)
       
        if diff:
            x_batches.append(last_batch)
            y_batches.append(last_ybatch)  
        
        losses=[]
        
        for epoch in range(1, self.epochs+1):
            
            print(f'Epoch {epoch}')
            plt.figure()
            print('Model Parameters : \n ' , self.hamiltonian.params)
            
            batch_errors = None
            
            
            batchnum = 1
            errors_epoch=[]
            
                                
            

            for i,x_batch in tqdm(enumerate(x_batches)):

                    
                    acc=self.predict_test(X_test,y_test,plot=plot)
                    #acc=predict_test_new(X,X_raw)
                    self.accuracy.append(acc)
                    #self.predict_test_new(X_norm,X_raw)
                    #visualize_samples(X_test, y_sel, sample_indices=list(range(12)), n_cols=6)
                    y_batch=y_batches[i]
                    errors = self.update_weights(x_batch, y_batch,learning_rate)
                    
                    if type(batch_errors) is np.ndarray:
                        batch_errors = np.hstack((batch_errors, errors))
                    else:
                        batch_errors = errors
                    #self.save_weights(
                        #f'e{epoch}_b{batchnum}_{self.paramstring}')
                    batchnum += 1
               
                    #self.save_weights(
                     #   f'e{epoch}_b{batchnum}_{self.paramstring}')
                    #raise e
                    errors_epoch.append(np.linalg.norm(errors))
            
            losses.append(errors_epoch)
            
            print(np.linalg.norm(self.weights))
            
            '''
            if save==True:
                try:
                    np.savez(f'./epoch{epoch}_weights_h{self.n_hidden_nodes}_v{self.dim_input}_lr{self.learning_rate}_e{self.epochs}',self.H.θ)
                    np.savez(f'./epoch{epoch}_losses_h{self.n_hidden_nodes}_v{self.dim_input}_lr{self.learning_rate}_e{self.epochs}',errors_epoch)
                except:
                    print('error_saving')
             '''
        
        
        return losses 
             



from hamiltonian import *
from GQEVT import *
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy
from sklearn.metrics import accuracy_score



class QBM:


    def __init__(self,n_hidden,n_visible,n_output,terms=None,connectivity='all',n=20,beta=0.1):
        
        self.hamiltonian=ModelHamiltonian(n_hidden,n_visible,terms=terms,n_output=n_output,connectivity='all')
        
        self.n=n
        self.beta=beta
        self.gqevt=GQEVT(n,-beta,'GQSP')
        #self.initial_params=initial_params
        #self.params=initial_params
        self.n_hidden=self.hamiltonian.n_hidden
        self.n_visible=self.hamiltonian.n_visible
        
        self.n_qubits=self.n_hidden+self.n_visible
        self.Ny=self.hamiltonian.Ny
        #self.n_y=1
        
        #self.n_x=n_visible-self.n_y
       

        self.weights= self.hamiltonian.get_weights()
       
    
    
    def update_model_params(self,weights):
        # reorder weights for simplicity
        
        i=0
        for key,value in self.hamiltonian.params.items():

            m=len(value)

            self.hamiltonian.params[key]=weights[i:i+m]

            i=i+1
            

        
    def _output_clamped_configs(self,x,y):
        
        
        
        self.H_x=self.hamiltonian.build_hamiltonians(x, y=None)
        self.H_xy=self.hamiltonian.build_hamiltonians(x, y)
        
    
        
        x_clamped = self._compute_expectation(self.H_x) 
        x_clamped=np.array(x_clamped)/x_clamped[0]   
            
        xy_clamped = self._compute_expectation(self.H_xy) 
        xy_clamped=np.array(xy_clamped)/xy_clamped[0]   
        
        
        
        return x_clamped,xy_clamped

        
       
        
        
            
    
    def _compute_expectation(self,H):
        
        
        
        
        new_wires=range(2,self.n_qubits+2)
        old_wires=range(0,self.n_qubits)
        
        
        H_new = H.map_wires(dict(zip(old_wires,new_wires)))
        proj = qml.Projector( [0] * 2,wires=range(0,2))
        new_ops=[proj]+[proj@op for op in H_new.ops]
        
        dev_xy = qml.device("default.qubit", wires=2*(len(H.wires))+2)
        self.gqevt.build(qml.matrix(H_new))
        @qml.qnode(dev_xy, interface="autograd")
        
        def _compute():

            
            for wire in range(2,len(H.wires)+2):
                qml.Hadamard(wire)
                qml.CNOT([wire,wire+len(H.wires)])
            
           
            self.gqevt.circuit(self.gqevt.angles,'direct_matrix')
            #return qml.expval(qml.PauliZ(0))
            #return qml.state()
            return [qml.expval(op) for op in new_ops]
            
        return _compute()    
        
        
    def update_weights(self,x_batch,y_batch,learning_rate):
        
        errors=0 
        
      
        for i,x_vector in enumerate(x_batch):
            
           
            y_vector=y_batch[i]
            y_vector=2*y_vector-1
            
           
            
            x_clamped,xy_clamped=self._output_clamped_configs(x_vector,y_vector)
            
            
            x_clamped=x_clamped[1:]
            
            xy_clamped=xy_clamped[1:]
            
            
            new_x_clamped,new_xy_clamped= self.hamiltonian.order_match_configs(x_clamped,xy_clamped,x_vector,y_vector,self.H_x.terms()[-1],self.H_xy.terms()[-1])
            
           
            
            errors+= (new_xy_clamped - new_x_clamped)

            
            
            
            
            
            

        errors /= x_batch.shape[0]
      
        self.weights = self.weights + learning_rate * errors
        #print('length of weight vector', len(self.weights))
        
        self.update_model_params(self.weights)
        
      
        
            
            
        return errors

    def predict(self,x):
        
        
        H_x=self.hamiltonian.build_hamiltonians(x, y=None)
       
        
    
        ops=[qml.PauliZ(i) for i in range(self.n_hidden,self.n_hidden+self.Ny)]
        
        x_clamped = self._compute_expectation(H_x) 
        
        x_clamped=np.array(x_clamped)/x_clamped[0]

        indices=[H_x.terms()[-1].index(op) for op in ops]

        

        output_vals=np.array([x_clamped[1+index] for index in indices])
        
       
        
        y_predict=(np.sign(output_vals)+1)/2
        
        prediction=0
        for i in range(len(y_predict)):
            prediction+=(2**i )* (y_predict[len(y_predict)-i-1])
        return output_vals,prediction
     
    def predict_test_new(self,X_test,y_test,plot=False):
        predictions=[] 
        for x_vector in X_test:
            
             exp_y,y_predict=self.predict(x_vector)
             predictions.append(y_predict)

       #
       
        acc_sklearn = accuracy_score(y_test, predictions)
        print(f"Accuracy (sklearn): {acc_sklearn:.4f}")

        
       
        #plt.axline((0, 0), slope=1, linestyle='--', color='gray', label="Decision boundary: x1 + x2 = 0")
        #plt.show()
        return acc_sklearn
    
    def predict_test(self,X_test,y_test):
        predictions=[] 
        for x_vector in X_test:
            
             exp_y,y_predict=self.predict(x_vector)
             predictions.append(y_predict)

        def to_cat(y_test):
            predictions=[]
            for y_predict in y_test:
                 prediction=0
                 for i in range(len(y_predict)):
                    prediction+=(2**i )* (y_predict[len(y_predict)-i-1])
                 predictions.append(prediction)
            return np.array(predictions)
        
        y_cat=to_cat(y_test)
        acc_sklearn = accuracy_score(y_cat, predictions)
        print(f"Accuracy (sklearn): {acc_sklearn:.4f}")
       
        
        plt.subplot(1,2,1)
        plt.title('Predictions')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='x', label="Test")

        plt.subplot(1,2,2)
        plt.title('Ground truth')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=to_cat(y_test), cmap='viridis', marker='x', label="Test")
        
        #plt.axline((0, 0), slope=1, linestyle='--', color='gray', label="Decision boundary: x1 + x2 = 0")
        plt.show()
        return acc_sklearn
    
    def train_model(self, X,y_data, X_test,y_test,batch_size=8, learning_rate=0.005,epochs=3,plot=False):
        self.accuracy=[]
        self.epochs=epochs
        if  len(X[0])+len(y_data[0])!= self.n_visible:
            raise ValueError(f" Insufficient visible nodes for dataset")
        
        data=X
      
        
        batch_num = data.shape[0] // batch_size
        
        diff = data.shape[0] % batch_size
        
        self.batch_size=batch_size
        
        if diff:
            
        
            data = data[:-diff]
            y_data= y_data[:-diff]
            last_batch = data[data.shape[0] - diff:]
            last_ybatch= y_data[y_data.shape[0] - diff:]
        
      
        x_batches = np.vsplit(data, batch_num)
     
        y_batches=np.vsplit(y_data, batch_num)
       
        if diff:
            x_batches.append(last_batch)
            y_batches.append(last_ybatch)  
        
        losses=[]
        
        for epoch in range(1, self.epochs+1):
            
            print(f'Epoch {epoch}')
            plt.figure()
            #print('Model Parameters : \n ' , self.hamiltonian.params)
            
            batch_errors = None
            
            
            batchnum = 1
            errors_epoch=[]
            
                                
            

            for i,x_batch in tqdm(enumerate(x_batches)):

                

                    #acc=self.predict_test(X_test,y_test)
                    acc=self.predict_test_new(X_test,y_test,plot)
                    self.accuracy.append(acc)
                    #self.predict_test_new(X_norm,X_raw)
                    #visualize_samples(X_test, y_sel, sample_indices=list(range(12)), n_cols=6)
                    y_batch=y_batches[i]
                    errors = self.update_weights(x_batch, y_batch,learning_rate)
                    
                    if type(batch_errors) is np.ndarray:
                        batch_errors = np.hstack((batch_errors, errors))
                    else:
                        batch_errors = errors
                    #self.save_weights(
                        #f'e{epoch}_b{batchnum}_{self.paramstring}')
                    batchnum += 1
               
                    #self.save_weights(
                     #   f'e{epoch}_b{batchnum}_{self.paramstring}')
                    #raise e
                    errors_epoch.append(np.linalg.norm(errors))
            
            losses.append(errors_epoch)
            
            print(np.linalg.norm(self.weights))
            
            '''
            if save==True:
                try:
                    np.savez(f'./epoch{epoch}_weights_h{self.n_hidden_nodes}_v{self.dim_input}_lr{self.learning_rate}_e{self.epochs}',self.H.θ)
                    np.savez(f'./epoch{epoch}_losses_h{self.n_hidden_nodes}_v{self.dim_input}_lr{self.learning_rate}_e{self.epochs}',errors_epoch)
                except:
                    print('error_saving')
             '''
        
        
        return losses 
             
    def visualize_final_model(self,X_norm,X_raw):
        predictions=[] 
        for x_vector in X_norm:
            
             exp_y,y_predict=self.predict(x_vector)
             predictions.append(y_predict)

       #
        visualize_samples(X_raw, y_predict, sample_indices=list(range(14)), n_cols=7)
        
