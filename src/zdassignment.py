"""
zd assignment
Yue Yu
yuey1@andrew.cmu.edu
"""
import sys
import os
def count_doc_number(path):
    """
    count the number of documents 
    in the directory
    @param path, the directory path
    @return the number of files in that path
    """
    if not os.path.isdir(path):
        return 0
    return len(os.listdir(path))    

if __name__ == "__main__":
    print(count_doc_number(sys.argv[1]))
