#Design Problem: Least Recently Used Cache
# capacity: fixed size cache
# get values from cache based on key
# put values in cache based on key --> if key exists, update value, doesn't exist put value in cache, if exceed the capacity, evict least recently used
# used in browsers

# get & put in O(1)
# cache as a hashmap to map key to node
# linked list to find out least & most recently used
#left=LRU, right=most used

class Node:

    def __init__(self, key, value) -> None:
        self.key, self.value = key, value
        self.prev = self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

        self.left, self.right = Node(0,0), Node(0,0)
        self.left.next=self.right
        self.right.prev=self.left
    
    ##helper functions

    # remove node from list
    def remove(self,node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    #insert node at right
    def insert(self,node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.next, node.prev = nxt, prev
    
    ## major functions

    # get value, and make the node as most recently used/accessed
    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].value
        return -1

    # update value if exists, else check if exceeds capacity, Y: delete LRU node, then put new node at right(most recent)
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        elif len(self.cache) == self.capacity:
            #remove from list & delete LRU from cache
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])      