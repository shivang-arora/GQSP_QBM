import itertools
import numpy as np
import pennylane as qml

operator_list = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}

class ModelHamiltonian:
    '''
    This class provides functionality to build arbitrary Hamiltonians for the QBM circuit.

    Attributes:
        
        n_hidden (int): Number of hidden qubits.
        n_visible (int): Number of visible qubits. (Equal to num_features+ log2(num_class_labels))
        
        Ny (int): Number of output qubits.
        
        terms (List[str]): Pauli terms to include in Hamiltonian (e.g., 'Z', 'ZZ').
        
        params (dict): Dictionary of trainable parameters per term.
        
        connections (dict): qubit combinations for each Pauli word.
    
    Example:
        
        >>> model = ModelHamiltonian(n_hidden=3, n_visible=4, terms=['Z', 'ZZ'])
        
        >>> hamiltonian = model.build_hamiltonians(x,y)

    '''
    
    def __init__(self, n_hidden: int, n_visible: int, terms: list[str] = None, n_output: int = 1, connectivity: str = 'all'):
        
        self.terms = terms if terms is not None else ['Z', 'ZZ']
        
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.Ny=n_output
        self.output_wires=range(self.n_hidden,self.n_hidden+self.Ny)
        if connectivity == 'all':
            # Add feature for limited connectivity
            self.generate_connections()
        
        self.initialize_params()
    
    
    def generate_connections(self):
        """
        Generates qubit connections as a dictionary for each term in the Hamiltonian.
            
            
        """
        
        qubits_hidden = ['h'+str(i) for i in range(self.n_hidden)]
        qubits_visible = ['v'+str(i) for i in range(self.n_visible)]
        qubits = qubits_hidden + qubits_visible
        self.connections = {}
        for term in self.terms:
            m = len(term)
            options = qubits if 'Z' in term else qubits_hidden
            combinations = list(itertools.combinations(options, m))
            self.connections[term] = combinations

    def initialize_params(self):
        self.params = {}
        for term in self.terms:
            self.params[term] = np.random.rand(len(self.connections[term]))

    def qubit_number(self, qubit):
        
        if qubit[0] == 'h':
            return int(qubit[1])
        else:
            return int(qubit[1]) + self.n_hidden

    def operators(self, term, connection):
        
        assert len(term) == len(connection)
        for i, op in enumerate(term):
            num = self.qubit_number(connection[i])
            
            if i == 0:
                a = operator_list[op](num)
            else:
                a = a @ operator_list[op](num)
        return a
    
    def fix_visible(self,connection,x,y):

        for label in connection:
            if 'v' in label:
                pass        
        
    
    def build_hamiltonians(self,x,y=None):
        """
            Build the parameterized Hamiltonian based on input (and optionally output) data.
            
            Args:
                x: array-like, visible unit configuration
                y: array-like or None, output unit configuration (class data)
                
            Returns:
                
                PennyLane Hamiltonian object
        """
       
        if y is None:        
            
            X=x
            
            reject_terms=['v'+str(i) for i in range(self.Ny,self.n_visible)]
            
        else:    
            X=np.concatenate((x,y))
            reject_terms=['v'+str(i) for i in range(self.n_visible)]
        
   
        coeffs, ops = [], []
    
        # Assigning pauli word for each connection, substituting values for visible and output qubits. 
      
        for pauli_term in self.terms:
         
            
            if 'Z' not in pauli_term:
                
                for i,coefficient in enumerate(self.params[pauli_term]):
                
                    coeffs.append(coefficient)
                    ops.append(self.operators(pauli_term,self.connections[pauli_term][i]))
        
            else:
                m=len(pauli_term)
               
                for i,connection in enumerate(self.connections[pauli_term]):
                         
                    factor=1
                    count=0
                    new_connection=[]
                    for label in connection:
                        
                        
                        if any(sub in label for sub in reject_terms):
                           
                           
                            factor*=X[int(label[1])-self.Ny]
                            count+=1
                           
                        else:
                            new_connection.append(label)
                            
                          
                                
                    
                    if count==0:
                        coeffs.append(self.params[pauli_term][i])
                        ops.append(self.operators(pauli_term,self.connections[pauli_term][i]))
                    else:
                        
                        new_term=pauli_term[count:]
                        
                        if new_term != '':    
                        
                            coeffs.append(self.params[pauli_term][i]*factor)
                           
                            
                            new_operator=self.operators(new_term,new_connection)
                            ops.append(new_operator)
                       
        
        hamiltonian=qml.Hamiltonian(coeffs,ops)        
          
        
        
        return hamiltonian.simplify()


    def remove_operators_on_wires(self,operator_terms, wires_to_remove):
        
        """Remove terms from an operator that act on any of the given wires."""
        
        #wires_to_remove = set(wires_to_remove)
        removed_wires_list = []
        new_ops = []
        
        for word in operator_terms:
            wires=word.wires.tolist()
            new_word_ops=[]
            new_word=word
           
            
            if any(x in wires_to_remove for x in wires):
                new_word= qml.Identity(wires[0])
               # removed_wires= list(set(wires) & set(wires_to_remove))
                
                for i,wire in enumerate(wires):
                   
                    
                    if wire not in wires_to_remove:
                      
                        new_word_ops.append(word[i])
                
            
                for i,k in enumerate(new_word_ops):
                    new_word=new_word@k
            
            removed_wires= list(set(wires) & set(wires_to_remove))
            
            new_ops.append(new_word.simplify())
    
            removed_wires_list.append(removed_wires)
        return new_ops,removed_wires_list
            
    
    def get_weights(self):
        
        """
            Retreive array of weights from the parameter dictionary .
            
            Args:
               
                
            Returns:
                
            nd.array
        """
        weights=np.array([])
        for term,values in self.params.items():
            weights=np.concatenate((weights,values))

        return weights
    
    def order_match_configs(self,x_clamped,xy_clamped,x,y,terms_x,terms_xy):

        
        """
            Match the and reorder expectation values for x-clamped and xy-clamped circuits, for evaluating the weights update vector.
            
            Args:
                x_clamped: array-like, expectation values for x-clamped 
                xy_clamped: array-like, expectation values for x-clamped  
                x: input data
                y: class data
            
            Returns:
            Tuple[np.ndarray, np.ndarray]:
                
             - Extended and ordered expectation values for x-clamped and xy-clamped circuits.
        
        """
        
        X=np.concatenate((y,x))
        self.visible_wires=range(self.n_hidden,self.n_hidden+self.n_visible)
        
        new_x_clamped=[]
        new_xy_clamped=[]
        for term,connections in self.connections.items():
            for connection in connections:
                
                operator=self.operators(term,connection)
            
                reduced_operator_x,removed_wires_x=self.remove_operators_on_wires([operator],self.output_wires)
                reduced_operator_xy, removed_wires_xy=self.remove_operators_on_wires([operator],self.visible_wires)
                prefactor_x=1
                prefactor_xy=1
               
                
                for wire in removed_wires_x[0]:
                    prefactor_x*= X[wire-self.n_hidden]
                for wire in removed_wires_xy[0]:
                    prefactor_xy*=X[wire-self.n_hidden]
               
            
                try:
                    index_x=terms_x.index(reduced_operator_x[0])
                   
                    value_x= x_clamped[index_x]*prefactor_x
                except:
                    index_x=None
                    value_x=prefactor_x
                
                try:
                    index_xy=terms_xy.index(reduced_operator_xy[0])
                    value_xy= xy_clamped[index_xy]*prefactor_xy
                
                except:
                    index_xy=None
                    value_xy=prefactor_xy

                new_x_clamped.append(value_x)
                new_xy_clamped.append(value_xy)
        return np.array(new_x_clamped),np.array(new_xy_clamped)
               
                
                
