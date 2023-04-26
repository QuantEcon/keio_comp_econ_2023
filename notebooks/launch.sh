jupytext --to notebook ../notebooks_source/*.md
mv ../notebooks_source/*.ipynb .
jupyter-notebook --no-browser --port=8080
