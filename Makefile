all: compile


compile:
	mkdir -p build && cd build && cmake ../csrc/liars_dice -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../cfvpy && make -j

compile_slow:
	mkdir -p build && cd build && cmake ../csrc/liars_dice -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../cfvpy && make -j2

test: | compile
	make -C build test
	nosetests cfvpy/

clean:
	rm -rf build cfvpy/rela*so
