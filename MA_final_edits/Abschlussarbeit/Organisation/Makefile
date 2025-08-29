
.PHONY: PDF

PDF: Abschlussarbeit.pdf

Abschlussarbeit.pdf: Abschlussarbeit.tex
	arara $<

clean:
	rm -f *.log
	rm -f *.aux
	rm -f *.out
	rm -f *.toc
	rm -f *.lot
	rm -f *.lof

purge: clean
	rm -f Abschlussarbeit.pdf
