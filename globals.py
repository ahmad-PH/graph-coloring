from io import TextIOWrapper
from utility import DataDump

data = DataDump()
debug_mode = False

def initialize_globals():
    global data
    data.file_writer = open('results.txt', 'w')
    
def free_globals():
    global data
    data.file_writer.close()