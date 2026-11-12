# Build the MPAT documentation:  make html | make fast | make linkcheck | make clean
SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?= -W --keep-going
SOURCEDIR    = source
BUILDDIR     = build

.PHONY: html fast linkcheck clean

html:       ## full build: renders the figures and animations that are out of date
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

fast:       ## text edits only: skip rendering (missing media become placeholders)
	MPAT_DOCS_ANIMATIONS=skip $(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

linkcheck:
	MPAT_DOCS_ANIMATIONS=skip $(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

clean:      ## remove all generated output, including rendered media and cached chains
	rm -rf $(BUILDDIR)
