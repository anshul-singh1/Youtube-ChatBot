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


## this is the code snppet to fix the notebook metadata, which was causing issues when running the notebook in a different environment. The "widgets" metadata is specific to the environment where the notebook was created and can cause errors when trying to run it elsewhere. By removing this metadata, we can ensure that the notebook runs smoothly in any environment.