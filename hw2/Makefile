all_execs = \
	val_test01_solved \
	val_test02_solved

all: $(all_execs)

val_test01_solved: val_test01_solved.cpp
	g++ -g $< -o $@

val_test02_solved: val_test02_solved.cpp
	g++ -g $< -o $@

clean:
	rm $(all_execs)
