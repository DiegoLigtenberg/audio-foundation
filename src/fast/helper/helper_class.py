
from collections import defaultdict

# Monkey-patching a subclass of defaultdict to add is_empty method 
class defaultdict(defaultdict):
    def is_empty(self):
        return isinstance(defaultdict)
    
    def contains(self,*keys):
        data = self
        for key in keys:
            if key not in data:
                return False
            data = data[key]  # move deeper into the nested structure
        return True
    
    def __repr__(self):
        # Return a string representation that looks like a normal dictionary
        return dict.__repr__(self)
 
# mydict = ({"metadata":[1,2,3]})
mydict = defaultdict(list)
mydict["technical_data"] = 5
print(mydict["technical_data"]) 