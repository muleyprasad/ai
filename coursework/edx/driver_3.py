# initialize goal state, visited states & frontier states
# add initial state to frontier

# do while frontier is not empty
    # take out first state out of frontier
    # add this state to visited
    # check if it's the goal state
        #if yes exit
        #if not expand
            # check generated nodes are not visited and not in frontier add them to frontier
      
#Test Case #1
#python driver.py bfs 3,1,2,0,4,5,6,7,8
#python driver.py dfs 3,1,2,0,4,5,6,7,8
#python driver.py ast 3,1,2,0,4,5,6,7,8

#Test Case #2
#python driver.py bfs 1,2,5,3,4,0,6,7,8
#python driver.py dfs 1,2,5,3,4,0,6,7,8
#python driver.py ast 1,2,5,3,4,0,6,7,8


from sys import *
from collections import OrderedDict, deque
import time
import resource
from queue import PriorityQueue

start_time = time.time()

def main():
    goal_state = [0,1,2,3,4,5,6,7,8]
    # validate inputs
    if len(argv) < 3:
        print('Invalid inputs, exiting ...')
        exit()
    else:
        algo = argv[1]
        start_state = list(map(int, argv[2].split(','))) 
        if(len(start_state) != 9):
            print('Invalid inputs, exiting ...')
            exit()
    
    if algo == 'bfs' or algo == 'dfs':
        solver(algo, start_state, goal_state)
    elif algo == 'ast':
        ast(start_state, goal_state)
    else:
        print('Invalid algo type, exiting ...', algo)
        exit()
        
    exit()
    
def solver(algo, start_state, goal_state):
    result_node = None
    isStack = algo == 'dfs'
    explored, frontier = NodeCollection(isStack), NodeCollection(isStack)
    # create goal state
    goal_node = Node(goal_state, -1, '', None)
    # add initial state to frontier
    frontier.put(Node(start_state,0,'', None))

    # do while frontier is not empty
    while len(frontier) > 0:
        # take out first state out of frontier
        current_node = frontier.get()
        # add this state to explored
        explored.put(current_node)
        # check if it's the goal state
        if(current_node == goal_node):
            #if yes stop processing and exit
            result_node = current_node
            break
        else:
            #if not expand
            # print('before expand logic',len(frontier))
            successors = filter(None, getSuccessors(current_node, algo))
            # check generated nodes are not visited and not in frontier add them to frontier
            for successor in successors:
                if successor not in frontier and successor not in explored:
                    frontier.put(successor) 
                    
    logOutput(result_node, explored, frontier)

def logOutput(input_node, explored, frontier):      
    f = open('output.txt','wt')
    
    if input_node is None:
        f.write("Unsolvable input..")
        return

    path_to_goal = deque()
    cost_of_path = 0
    result_node = input_node
    while input_node.parent is not None:
        path_to_goal.appendleft(input_node.action)
        input_node = input_node.parent
        cost_of_path += 1

    f.write("path_to_goal: " + str(list(path_to_goal)) + "\n")
    f.write("cost_of_path: "+ str(cost_of_path) + "\n")
    f.write("nodes_expanded: " + str(len(explored) - 1) + "\n")  #result node wasn't expanded
    f.write("search_depth: "+ str(result_node.depth) + "\n")
    f.write("max_search_depth: "+ str(frontier.maxdepth if frontier.maxdepth > explored.maxdepth else explored.maxdepth) + "\n")
    f.write("running_time: "+ format(time.time() - start_time) + "\n")
    f.write("max_ram_usage: " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024))

def getSuccessors(current_node, algo):
    if algo == 'bfs' or algo == 'ast':
        return [getSuccessor(current_node,'U',algo),getSuccessor(current_node,'D',algo),
                getSuccessor(current_node,'L',algo),getSuccessor(current_node,'R',algo)]
    elif algo == 'dfs':
        return [getSuccessor(current_node,'R',algo),getSuccessor(current_node,'L',algo),
                getSuccessor(current_node,'D',algo),getSuccessor(current_node,'U',algo)]
    else:
        return []
                
def getSuccessor(node, action,algo):
    successor = None
    # find blank_space_loc
    b = node.state.index(0)
    bRow = b // 3
    bCol = b % 3
    if action == 'U' and b > 2:
        successor = buildSuccessor(node, b - 3, b, 'Up',algo)
    elif action == 'D' and b < 6:
        successor = buildSuccessor(node, b + 3, b, 'Down',algo)
    elif action == 'L' and b not in [0,3,6]:
        successor = buildSuccessor(node, b - 1, b, 'Left',algo)
    elif action == 'R' and b not in [2,5,8]:
        successor = buildSuccessor(node, b + 1, b, 'Right',algo)
    
    # if action == 'U' and bRow < 2:
    #     successor = buildSuccessor(node, b + 3, b, 'Up',algo)
    # elif action == 'D' and bRow > 0:
    #     successor = buildSuccessor(node, b - 3, b, 'Down',algo)
    # elif action == 'L' and bCol < 2:
    #     successor = buildSuccessor(node, b + 1, b, 'Left',algo)
    # elif action == 'R' and bCol > 0:
    #     successor = buildSuccessor(node, b - 1, b, 'Right',algo)

    return successor

def buildSuccessor(node, swapIndex, blankIndex, action,algo):
    successorState = list(node.state)
    # swap values to take an action  
    successorState[swapIndex], successorState[blankIndex] = successorState[blankIndex], successorState[swapIndex]
    if algo == 'ast':
        successor = NodeH(successorState, node.depth + 1, action, node)
    else:
        successor = Node(successorState, node.depth + 1, action, node)
    return successor

class Node:
    def __init__(self, state, depth, action, parent):
        self.key = int("".join([str(s) for s in state]))
        self.state = state
        self.depth = depth
        self.action = action
        self.parent = parent
        
    def __eq__(self, other):
        return self.state == other
    
    def __str__(self):
        return str(self.state) + str(self.depth) + self.action
        
class NodeCollection:
    def __init__(self, isStack):
        self.q = OrderedDict()
        self.isStack = isStack
        self.maxdepth = 0
        
    def put(self, node):
        if self.maxdepth < node.depth:
            self.maxdepth = node.depth
            
        self.q[node.key] = node
        
    def get(self):
        key, val = self.q.popitem(self.isStack)
        return val

    def __contains__(self, node):
        return node.key in self.q

    def __len__(self):
        return len(self.q)
        
    def __getitem__(self, index):
        keys = list(self.q.keys())
        # print(len(keys))
        # print(index)
        if len(keys) > index:
            return self.q[keys[index]]

class NodeH(Node):
    def __init__(self, state, depth, action, parent):
        Node.__init__(self, state, depth, action, parent)
        gN = parent.depth if parent else 0
        self.heuristic = self.calcHeuristic(state) + gN
        
    def calcHeuristic(self, state):
        h = 0
        for i in range(0,len(state)):
            if state[i] != 0:
                # state[i] value is our expected index
                # i is our current index
                goalRow = state[i] // 3
                goalCol = state[i] % 3
                curRow = i // 3
                curCol = i % 3
                v = abs(goalRow - curRow) + abs(goalCol - curCol)
                h += v
        return h
        
    def __lt__(self, other):
        return self.heuristic < other.heuristic
        
    def __str__(self):
        return str(self.heuristic) + ' ' + str(self.state) + str(self.depth) + self.action
        
class PriorityNodeCollection:
    def __init__(self):
        self.q = PriorityQueue()
        self.maxdepth = 0
        
    def put(self, node):
        if self.maxdepth < node.depth:
            self.maxdepth = node.depth
        self.q.put(node)
        
    def get(self):
        return self.q.get()

    def __contains__(self, node):
        return node in self.q.queue

    def __len__(self):
        return len(self.q.queue)
        
    # def __getitem__(self, index):
    #     keys = list(self.q.keys())
    #     # print(len(keys))
    #     # print(index)
    #     if len(keys) > index:
    #         return self.q[keys[index]]

def ast(start_state, goal_state):
    result_node = None
    explored, frontier = PriorityNodeCollection(), PriorityNodeCollection()
    # create goal state
    goal_node = NodeH(goal_state, -1, '', None)
    # add initial state to frontier
    frontier.put(NodeH(start_state,0,'', None))

    # do while frontier is not empty
    while len(frontier) > 0:
        # take out first state out of frontier
        current_node = frontier.get()
        # add this state to explored
        explored.put(current_node)
        # check if it's the goal state
        if(current_node == goal_node):
            #if yes stop processing and exit
            result_node = current_node
            break
        else:
            #if not expand
            successors = filter(None, getSuccessors(current_node, 'ast'))
            # check generated nodes are not visited and not in frontier add them to frontier
            for successor in successors:
                if successor not in frontier and successor not in explored:
                    frontier.put(successor) 

    logOutput(result_node, explored, frontier)
    
if __name__ == "__main__":
	main()



