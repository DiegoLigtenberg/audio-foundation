from collections import defaultdict

# Create a defaultdict with a lambda that returns a dictionary with 'count' and 'sum' as default values
my_dict = defaultdict(lambda: {'count': 0, 'sum': 0})

# Adding values to the dictionary
my_dict['a']['count'] += 1
my_dict['a']['sum'] += 10

my_dict['b']['count'] += 1
my_dict['b']['sum'] += 5

my_dict['c']

print(my_dict)  
# Output: defaultdict(<function <lambda> at 0x...>, {'a': {'count': 1, 'sum': 10}, 'b': {'count': 1, 'sum': 5}})
