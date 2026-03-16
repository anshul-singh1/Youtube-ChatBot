import nbformat

files = [
"YTchatbot_Colab.ipynb"
]

for f in files:
    nb = nbformat.read(f, as_version=4)
    
    if "widgets" in nb["metadata"]:
        del nb["metadata"]["widgets"]
        
    nbformat.write(nb, f)

print("Fixed notebook metadata")