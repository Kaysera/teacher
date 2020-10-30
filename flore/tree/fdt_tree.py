import pandas as pd
import skfuzzy as fuzz
import math
import numpy as np
from numpy import ma
from scipy.stats import entropy
from flore.fuzzy import fuzzy_entropy, weighted_fuzzy_entropy

class TreeFDT:
    def __init__(self, features):
        self.var_index = -1 # TODO: REMOVE WHEN CHANGE TO NEW PREDICT
        self.is_leaf = True
        self.childlist = []
        self.splits = [] # TODO: REMOVE WHEN CHANGE TO NEW PREDICT
        self.class_value = -1
        self.level = -1
        self.error = 0
        self.value = (0,0)
        
    def __str__(self):
        output = '\t'*self.level 
        if(self.is_leaf):
            output += 'Class value: ' + str(self.class_value)
        else:
            output += 'Feature '+ str(self.value)
            for child in self.childlist:
              output+='\n'
              output+= '\t'*self.level
              output+= 'Feature '+ str(child.value)
              output += '\n'+str(child)
            output += '\n'+'\t'*self.level
        return output
    
    def predict(self,x):
        if self.is_leaf:
            return self.class_value
        else:
            i=0
            for spl in self.splits:
                if x[self.var_index] == spl:
                    return self.childlist[i].predict(x)
                else: i = i+1

class FDT:
    def __init__(self, features, fuzzy_set_df, fuzzy_threshold=0.0001, th=0.0001, max_depth=10, min_num_examples = 1, prunning=True):
        self.max_depth = max_depth
        self.tree = TreeFDT(features)
        self.features = features
        self.min_num_examples = min_num_examples
        self.prunning = prunning
        self.th = th
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_set_df = fuzzy_set_df

        
    def fit(self, X, y):
        self.partial_fit(X, y, ma.masked_all(len(y)).mask, self.fuzzy_set_df, self.tree, 0)

    def get_max_f_gain(self,y,fuzzy_set_df):
        best_att = ''
        best_f_gain = 0
        for feature in fuzzy_set_df:
            f_ent = 0
            for triangle in fuzzy_set_df[feature]:
                verbose = False
                if feature == 'theory':
                    verbose = True
                f_ent += fuzzy_entropy(fuzzy_set_df[feature][triangle], y.to_numpy())
            f_gain = f_ent - weighted_fuzzy_entropy(fuzzy_set_df[feature], y)
            if f_gain > best_f_gain:
                best_f_gain = f_gain
                best_att = feature

        return(best_att, best_f_gain)

    def stop_met(self, f_gain, y_masked, level):
        if len(np.unique(y_masked)) < 2:
            return True
        if len(y_masked) < self.min_num_examples:
            return True
        if level >= self.max_depth:
            return True
        if f_gain < self.fuzzy_threshold:
            return True

    def get_class_value(self, att_masked, y_masked):
        cv = {}
        # TODO: PREGUNTAR COMO SE OBTIENE EL VALOR CLASE
        # AHORA MISMO ES LA CARDINALIDAD QUE CUENTA SOLO 
        # EL VALOR DEL ATRIBUTO DEL NODO
        for class_value in np.unique(y_masked):
            mask = y_masked == class_value
            cv[class_value] = (att_masked * mask).sum() / att_masked.sum()
        return cv



    
    def partial_fit(self, X, y, mask, fuzzy_set_df, current_tree, current_depth):
        current_tree.level = current_depth
        att, f_gain = self.get_max_f_gain(y, fuzzy_set_df)
        print(current_tree.value)
        # apply mask to y
        y_masked = y.to_numpy()[mask]
        print(y.to_numpy()[mask])

        if self.stop_met(f_gain, y_masked, current_depth):
            current_tree.is_leaf = True

            if current_tree.value == (0,0):
                # CASO NODO RAIZ
                current_tree.class_value = 0
            else:
                X_masked = self.fuzzy_set_df[current_tree.value[0]][current_tree.value[1]] 
                X_masked = X_masked[mask]
                # X_masked ES EL CONJUNTO BORROSO DEL ATRIBUTO Y VALOR AL QUE PERTENECE EL NODO
                current_tree.class_value = self.get_class_value(X_masked, y_masked)
                print(current_tree.class_value)
            
            return

        current_tree.is_leaf = False
        for value in fuzzy_set_df[att]:
            new_mask = np.ma.masked_greater(fuzzy_set_df[att][value], 0)
            sz = mask * new_mask.mask
            new_fuzzy_set_df = fuzzy_set_df.copy()
            del(new_fuzzy_set_df[att])
            child = TreeFDT(self.features)
            child.value = (att, value)
            current_tree.childlist.append(child)
            self.partial_fit(X, y, sz, new_fuzzy_set_df, child, current_depth+1)
