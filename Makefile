TEX = xelatex
FLAGS = -interaction=nonstopmode -halt-on-error

# All standalone documents (report class, shared preamble)
PARTS = main \
        part0_camera_math \
        part0_colmap_theory \
        part0_progress_report \
        part2_motion_analysis \
        part3_pitching \
        part4_data_analysis \
        part4_openpose_smpl_fitting \
        part5_constraints_setup \
        part5_dev_setup \
        part5_research_proposal \
        fitting_guide \
        methodology_survey \
        vitpose_cam3_analysis \
        plan_7view_full

all: $(addsuffix .pdf, $(PARTS))

%.pdf: %.tex preamble/*.tex
	$(TEX) $(FLAGS) $<
	$(TEX) $(FLAGS) $<

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk

.PHONY: all clean
