# import matplotlib.backends
# import os.path

# def is_backend_module(fname):
#     """Identifies if a filename is a matplotlib backend module"""
#     return fname.startswith('backend_') and fname.endswith('.py')

# def backend_fname_formatter(fname): 
#     """Removes the extension of the given filename, then takes away the leading 'backend_'."""
#     return os.path.splitext(fname)[0][8:]

# # get the directory where the backends live
# backends_dir = os.path.dirname(matplotlib.backends.__file__)

# # filter all files in that directory to identify all files which provide a backend
# backend_fnames = filter(is_backend_module, os.listdir(backends_dir))

# backends = [backend_fname_formatter(fname) for fname in backend_fnames]

# print (backends)
import matplotlib
matplotlib.use("pdf")