.PHONY: xml all %.xpdf handouts
.PRECIOUS: %.pdf

weeks := 1 # 2 3 4 5 6 7 # 8
#weeks := 1 2 3 4 5 6 7 # 8
#weeks := 1 2 3 4 5 6 7 8
#xpdf  := mupdf

all: $(foreach n, $(weeks), only_week$(n).xbundle) welcome.xbundle # only_intro.xbundle only_part2.xbundle 
xml: ../course.xml

%/course.xml: intro.tex sfmx.tex $(foreach n, $(weeks), week$(n).tex)
	latex2edx -d $(*) $<
	#./RELOAD

# section files are named like week1_4_hw.tex
only_%.xbundle: %.tex only_%.tex
	# latex2edx -P -d .. -m only_$(*).tex
	latex2edx -d .. --output-course-unit-tests $(*)_tests.yaml -m only_$(*).tex
	# ./RELOAD

welcome.xbundle: welcome.tex course_intro.tex
	latex2edx -d .. -m welcome.tex
	cp ../html/About_this_course.xml ../tabs/syllabus.html

# only_week1.xbundle: week1_[1-9]_*.tex week1_lectures.tex week1_lectures2.tex

only_week1.xbundle: week1_*.tex week1_lectures*.tex week1.tex

only_week2.xbundle: week2_[1-9]_*.tex week2_lectures.tex week2_lectures2.tex

only_week3.xbundle: week3_[1-9]_*.tex week3_lectures.tex week3_lectures2.tex

only_week4.xbundle: week4_[0-9]_*.tex week4_lectures.tex week4_lectures2.tex 

# dnd/wk4-grover1_dnd.xml dnd/wk4-grover2_dnd.xml dnd/wk4-grover-rth_dnd.xml dnd/wk4-grover-exact_dnd.xml 

only_week5.xbundle: week5_[1-9]_*.tex week5_lectures.tex week5_lectures2.tex week5_lectures3.tex dnd/wk4-grover-exact2_dnd.xml week5_2_grover.tex

only_week6.xbundle: week6_[0-9]_*.tex week6_lectures.tex week6_lectures2.tex week6_lectures3.tex

only_week7.xbundle: week7_[1-9]_*.tex week7_lectures.tex 

only_part2.xbundle: part2.tex part2_*.tex

%_dnd.xml: %.tex
	cd $(dir $<) && latex2dnd $(notdir $<) -r 210 -v -C
	mkdir -p ../static/images/$(notdir $(*))/
	cp -p $(*)*.png ../static/images/$(notdir $(*))/

check_policy:
	cd ../policies/2017_Fall;	python check_policy.py
