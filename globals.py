from io import TextIOWrapper

class DataDump:
	def __init__(self):
		object.__setattr__(self, "data", {})

	def __setattr__(self, name, value):
		self.data[name] = value

	def __getattr__(self, name):
		return self.data[name]

data = DataDump()

def initialize_globals():
    global data
    data.file_writer = open('results.txt', 'w')
    
def free_globals():
    global data
    data.file_writer.close()