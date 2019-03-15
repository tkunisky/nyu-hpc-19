# Set a compiler variable to be able to pass in the arguments needed
# for using OpenMP with Apple Clang.
OMP_COMPILER = g++ -Xpreprocessor -fopenmp -lomp
# OMP_COMPILER = g++ -fopenmp 

ALL_EXECS = \
	val_test01_solved \
	val_test02_solved \
	omp_solved2 \
	omp_solved3 \
	omp_solved4 \
	omp_solved5 \
	omp_solved6

all: $(ALL_EXECS)

val_test01_solved: val_test01_solved.cpp
	g++ -g $< -o $@

val_test02_solved: val_test02_solved.cpp
	g++ -g $< -o $@

omp_solved2: omp_solved2.cpp
	$(OMP_COMPILER) -g $< -o $@

omp_solved3: omp_solved3.cpp
	$(OMP_COMPILER) -g $< -o $@

omp_solved4: omp_solved4.cpp
	$(OMP_COMPILER) -g $< -o $@

omp_solved5: omp_solved5.cpp
	$(OMP_COMPILER) -g $< -o $@

omp_solved6: omp_solved6.cpp
	$(OMP_COMPILER) -g $< -o $@

clean:
	rm $(ALL_EXECS)
