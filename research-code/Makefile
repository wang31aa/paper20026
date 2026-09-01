PYTHON ?= python3

.PHONY: verify full-rebuild figures panel-provenance manuscripts submission-bundle package-check clean

verify:
	./reproduce.sh

full-rebuild:
	FULL_REBUILD=1 ./reproduce.sh

figures:
	MPLBACKEND=Agg $(PYTHON) figures/build_fig1_discovery.py
	MPLBACKEND=Agg $(PYTHON) cross_domain_v10/plot_v10.py
	MPLBACKEND=Agg $(PYTHON) boundary_law/derive_boundary_law.py
	$(PYTHON) boundary_law/validate.py
	MPLBACKEND=Agg $(PYTHON) discovery_extension/run_extension.py
	$(PYTHON) discovery_extension/validate.py
	$(PYTHON) figures/build_figures.py
	$(PYTHON) figures/build_observer_figure.py
	$(PYTHON) figures/build_matlab_reproduction_figure.py
	$(PYTHON) figures/build_cps_transfer_figure.py
	$(PYTHON) figures/build_openmct_figure.py
	$(PYTHON) figures/build_observer_in_loop_certified_figure.py
	$(PYTHON) figures/build_r4_mechanism.py
	$(PYTHON) figures/build_openmct_greybox_bridge.py
	$(PYTHON) tools/validate_observer_in_loop_figure.py
	$(PYTHON) tools/build_panel_provenance.py
	$(PYTHON) tools/validate_panel_provenance.py
	$(PYTHON) tools/validate_figure_fonts.py

panel-provenance:
	$(PYTHON) tools/build_panel_provenance.py
	$(PYTHON) tools/validate_panel_provenance.py

submission-bundle:
	$(PYTHON) tools/build_submission_bundle.py
	cd submission_bundle && shasum -a 256 -c SHA256SUMS.txt

manuscripts:
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error NC_Rebuilt_Main.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error NC_Rebuilt_Main.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error NC_Rebuilt_SI.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error NC_Rebuilt_SI.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error NC_Rebuilt_Extended_Data.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error NC_Rebuilt_Extended_Data.tex

package-check:
	$(PYTHON) tools/validate_package_manifest.py package_manifest.json

clean:
	find manuscript -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' \) -delete
