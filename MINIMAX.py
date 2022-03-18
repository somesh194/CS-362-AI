#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time


# In[2]:


class Node():
    
    def __init__(self, state, depth, maximiser, player, parent=None,):
        self.state = state
        self.children = list()
        self.score = 0
        self.depth = depth
        self.maximiser = maximiser
        self.player = player
        self.best_child = None
        
        if parent is not None:
            parent.children.append(self)
        
    def __hash__(self):
        
        return hash(str(self.state))
    
    


# In[3]:


class Environment():
    
    def __init__(self, start_state=None):
        
        if start_state is None:
            self.start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
        else:
            self.start_state = start_state
    def get_moves(self, state, player):
        
        new_states = []
        spaces = []
        for i in range(3):
            for j in range(3):
                if state[i][j]=='.':
                    new_state = state.copy()
                    new_state[i,j] = player
                    new_states.append(new_state)
        
        return new_states
    
    def check_terminal(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j]=='.':
                    return False
        
        return True
    
    def evaluate(self, state):
        
        for val in range(3):
            if state[val,0] == state[val,1] == state[val,2]!='.':
                if state[val, 0]=='x':
                    return 1
                else:
                    return -1
            
            if state[0,val] == state[1,val] == state[2,val]!='.':
                if state[0,val]=='x':
                    return 1
                else:
                    return -1
        
        if state[0,0] == state[1,1] == state[2,2]!='.':
            if state[0,0]=='x':
                return 1
            else:
                return -1
        
        if state[0,2] == state[1,1] == state[2,0]!='.':
            if state[0,2]=='x':
                return 1
            else:
                return -1
        
        return 0

    def get_start_state(self):
        return self.start_state


# In[4]:


class Agent():
    
    def __init__(self, env, maximiser):
        
        self.env = env
        self.start_state = env.get_start_state()
        self.root_node = None
        self.neginf = -10**18
        self.posinf = 10**18
        self.maximiser = maximiser
        self.player = 'x' if maximiser else 'o';
        self.explored_nodes = 0
    
    def minimax(self, node):
        
        self.explored_nodes+=1
        score = self.env.evaluate(node.state)
        if score!=0:
            node.score = score
            return node
        
        if self.env.check_terminal(node.state):
            node.score = 0
            return node
        
        if node.maximiser:
            
            best_score = self.neginf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1, 
                             maximiser=not node.maximiser, player='o', parent=node)
                
                child= self.minimax(child)
                node.children.append(child)
                
                if best_score<child.score or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
            
            node.depth = best_depth
            node.score = best_score
            
            return node
        
        else:
            best_score = self.posinf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1, 
                             maximiser=not node.maximiser, player='x', parent=node)
                
                child = self.minimax(child)
                node.children.append(child)
                
                if best_score>child.score  or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
            
            node.depth = best_depth
            node.score = best_score
            
            return node

    def run(self):
        
        self.root_node = Node(state=self.start_state, depth=0, maximiser=self.maximiser,
                             player=self.player, parent=None)
        
        self.root_node = self.minimax(node = self.root_node)
        
    def print_nodes(self):
        
        node = self.root_node
        
        while node is not None:
            print(node.state)
            node = node.best_child


# In[5]:


t = 0
for i in range(10):
    start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
    env = Environment(start_state = start_state)
    agent = Agent(env, maximiser=True)
    start_time = time.time()
    agent.run()
    end_time = time.time()
    t += end_time-start_time
print("Average time required:", t/10)


# In[6]:


agent.print_nodes()


# In[7]:


agent.explored_nodes


# In[8]:


class AlphaBetaAgent():
    
    def __init__(self, env, maximiser):
        
        self.env = env
        self.start_state = env.get_start_state()
        self.root_node = None
        self.neginf = -10**18
        self.posinf = 10**18
        self.maximiser = maximiser
        self.player = 'x' if maximiser else 'o';
        self.explored_nodes = 0
    
    
    def minimax(self, node, alpha, beta):
        
        self.explored_nodes+=1
        score = self.env.evaluate(node.state)
        if score!=0:
            node.score = score
            return node
        
        if self.env.check_terminal(node.state):
            node.score = 0
            return node
        
        if node.maximiser:
            
            best_score = self.neginf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1, 
                             maximiser=not node.maximiser, player='o', parent=node)
                
                child= self.minimax(node = child, alpha=alpha, beta=beta)
                node.children.append(child)
                
                if best_score<child.score or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            node.depth = best_depth
            node.score = best_score
            
            return node
        
        else:
            best_score = self.posinf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1, 
                             maximiser=not node.maximiser, player='x', parent=node)
                
                child = self.minimax(node = child, alpha=alpha, beta=beta)
                node.children.append(child)
                
                if best_score>child.score  or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
                
                beta = min(beta, best_score)
                if beta<=alpha:
                    break
            
            node.depth = best_depth
            node.score = best_score
            
            return node

    def run(self):
        
        self.root_node = Node(state=self.start_state, depth=0, maximiser=self.maximiser,
                             player=self.player, parent=None)
        
        self.root_node = self.minimax(node = self.root_node, alpha = self.neginf, beta = self.posinf)
        
    def print_nodes(self):
        
        node = self.root_node
        
        while node is not None:
            print(node.state)
            node = node.best_child


# In[9]:


start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
env = Environment(start_state = start_state)
agent = AlphaBetaAgent(env, maximiser=True)
agent.run()
agent.print_nodes()


# In[10]:


import time


# In[11]:


t = 0
for i in range(10):
    start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
    env = Environment(start_state = start_state)
    agent = AlphaBetaAgent(env, maximiser=True)
    start_time = time.time()
    agent.run()
    end_time = time.time()
    t += end_time-start_time
print("Average time required:", t/10)


# In[12]:


agent.print_nodes()


# In[13]:


agent.explored_nodes
