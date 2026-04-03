import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print('DEBUG: httpx version at test start:', __import__('httpx').__version__)
