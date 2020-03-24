all:
	latexmk -pdf main.tex
	echo "Document status: typeset"
